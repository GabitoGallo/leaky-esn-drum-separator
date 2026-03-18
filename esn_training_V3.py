import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.linear_model import Ridge
import librosa
import gc
import os
import optuna

# --- 1. CONFIGURAZIONE E PARAMETRI ---
FILENAME_MODELLO = "Modelli/isola_drums.npz"
FILENAME_OTTIMIZZAZIONE = "message in a bottle"
TRAIN_MODE = True  # <--- Metti FALSE per usare il modello salvato senza riaddestrare
OPTIMIZE_MODE = False # <--- Metti TRUE per eseguire l'ottimizzazione dei parametri con Optuna

# Parametri ESN   # Risultati di un ottimizzazione di 50 cicli: Migliori parametri trovati: {'rho': 0.9394467199557968, 'leak_rate': 0.9987669596415398, 'input_scale': 0.015948970814106123, 'alpha': 0.8609864619440879, 'sparsity': 0.2973828218165347}
n_units = 3000      
rho = 0.8       
a = 0.999           
input_scale = 0.0159
density = 0.297

# Parametro di Fit
Ridge_alpha = 100

# Parametri Audio
n_fft = 2048
hop_length = 512
n_freq = n_fft // 2 + 1 

# Dataset e segmenti
file_segments = {
    '2222': (0, None) ,             #(30, 100)
    'luna': (0, None) ,              #(15, 80)         
    'sensation': (0, None) ,        #(60, 120)
    'spectators': (0, None) ,      #(170, 220)
    'voli': (0, None) ,             #(60, 120)
    'becca': (0, None)             #(0, None)
}

# --- 2. FUNZIONI CORE ---

def get_states(inputs, n_units, W_in, W_res, leak_rate):
    """ESN Bidirezionale: Guarda passato e futuro."""

    inputs_gpu = cp.asarray(inputs)
    W_in_gpu = cp.asarray(W_in)
    W_res_gpu = cp.asarray(W_res)

    n_samples = len(inputs)
    states_fwd = cp.zeros((n_samples, n_units))
    x = cp.zeros(n_units)
    
    # 1. Passaggio in AVANTI
    states_fwd = cp.zeros((n_samples, n_units))
    x = cp.zeros(n_units)
    for t in range(n_samples):
        innovation = cp.tanh(cp.dot(W_in_gpu, inputs_gpu[t]) + cp.dot(W_res_gpu, x))
        x = (1 - leak_rate) * x + leak_rate * innovation
        states_fwd[t, :] = x
        
    # 2. Passaggio all'INDIETRO (Flip dell'input)
    inputs_rev = cp.flip(inputs_gpu, axis=0)
    states_bwd = cp.zeros((n_samples, n_units))
    x = cp.zeros(n_units)
    for t in range(n_samples):
        innovation = cp.tanh(cp.dot(W_in_gpu, inputs_rev[t]) + cp.dot(W_res_gpu, x))
        x = (1 - leak_rate) * x + leak_rate * innovation
        states_bwd[t, :] = x
        
    # Giriamo di nuovo il backward per allinearlo
    states_bwd = cp.flip(states_bwd, axis=0)
    
    # 3. CONCATENIAMO (Ora ogni istante ha 2 * n_units informazioni)
    return cp.asnumpy(cp.hstack((states_fwd, states_bwd)))

def init_weights(n_units, n_freq, rho, a, input_scale, seed=42, sparsity=density):
    """Inizializza le matrici casuali del reservoir."""
    print("Inizializzazione Reservoir...")
    np.random.seed(seed)
    np.random.seed(42)
   
    mask = np.random.uniform(-1, 1, (n_units, n_units)) < sparsity  # Moltiplica: dove c'è False diventa 0, dove c'è True resta il valore
    W_res = np.random.uniform(-1, 1, (n_units, n_units)) * mask

    W_in = np.random.uniform(-1, 1, (n_units, n_freq)) * input_scale 

    # Calibrazione raggio spettrale (effettivo)
    # Nota: Questo calcolo può essere lento per n_units > 2000
    Jacob = a * W_res + (1 - a) * np.eye(n_units)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(Jacob))) 
    W_res *= (rho / spectral_radius)
    
    return W_in, W_res

def train_model(file_segments, W_in, W_res, n_units, a, ridge_alpha):
    """Esegue il ciclo di training sui file e restituisce W_out."""
    all_states = []
    all_targets = []

    print("-" * 30)
    print("INIZIO TRAINING")
    
    for name, (start_sec, end_sec) in file_segments.items():
        print(f"Processing: {name}...")
        
        # Caricamento
        try:
            fs, mix = wavfile.read(f'Dataset/{name}_mix.wav')
            _, drums = wavfile.read(f'Dataset/{name}_drum.wav')
        except FileNotFoundError:
            print(f"Errore: File {name} non trovato. Salto.")
            continue

        # Selezione spezzone
        start_sample = int(start_sec * fs)
        end_sample = int(end_sec * fs) if end_sec is not None else len(mix)
        mix = mix[start_sample:end_sample]
        drums = drums[start_sample:end_sample]

        # Pre-processing
        if len(mix.shape) > 1: mix = np.mean(mix, axis=1)
        if len(drums.shape) > 1: drums = np.mean(drums, axis=1)
        
        mix = mix.astype(np.float32) / (np.max(np.abs(mix)) + 1e-9)
        drums = drums.astype(np.float32) / (np.max(np.abs(drums)) + 1e-9)

        # STFT
        F_mix = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)
        F_drums = librosa.stft(drums, n_fft=n_fft, hop_length=hop_length)
        mod_mix, _ = librosa.magphase(F_mix)
        mod_drums, _ = librosa.magphase(F_drums)

        # Stati del reservoir
        u = np.log1p(mod_mix.T)



        states = get_states(u, n_units, W_in, W_res, a).astype(np.float32)
        
        all_states.append(states)
        all_targets.append(np.log1p(mod_drums.T).astype(np.float32))

        # Pulizia RAM
        del mix, drums, F_mix, F_drums, mod_mix, mod_drums, u, states
        gc.collect()

    # Stacking finale
    print("Concatenazione matrici...")
    X_total = np.vstack(all_states)
    Y_total = np.vstack(all_targets)
    
    del all_states, all_targets
    gc.collect()

    # Ridge Regression
    print(f"Fitting Ridge Regression su shape {X_total.shape}...")
    ridge_model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge_model.fit(X_total, Y_total)
    W_out = ridge_model.coef_
    
    print("Training completato.")
    return W_out

def separa(nome_file, W_in, W_res, W_out, a, directory ="Brani Input"):
    """Separa i drums da un nuovo file usando le matrici fornite."""
    print(f"\nSeparazione in corso per: {nome_file}...")
    
    # Carica file target
    fs, mix = wavfile.read(f'{directory}/{nome_file}')
    if len(mix.shape) > 1: mix = np.mean(mix, axis=1)
    
    mix = mix.astype(np.float32)
    mix /= np.max(np.abs(mix)) + 1e-9

    # STFT
    F_mix = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)
    mod_mix, phase_mix = librosa.magphase(F_mix)
    
    # ESN Forward Pass
    u = np.log1p(mod_mix.T)



    X_states = get_states(u, n_units, W_in, W_res, a)
    
    # Predizione
    drums_pred_log = X_states @ W_out.T

    max_log_mix = np.max(u) 
    drums_pred_log = np.clip(drums_pred_log, a_min=None, a_max=max_log_mix)
    drums_pred_mod = np.expm1(drums_pred_log)
    
    # Ricostruzione Audio
    # Assicuriamo che non ci siano valori negativi nell'ampiezza
    drums_pred_mod = np.maximum(drums_pred_mod, 0)
    
    drums_pred = drums_pred_mod.T * phase_mix
    drums_out = librosa.istft(drums_pred, hop_length=hop_length)
    
    max_val = np.max(np.abs(drums_out))
    if max_val > 0:
        drums_out = drums_out / max_val * 0.9 
  
    
    # Salvataggio
    output_name = f'Tracce Separate/drums_esn_{nome_file}'
    wavfile.write(output_name, fs, (drums_out * 32767).astype(np.int16))
    print(f"Salvato: {output_name}")

    return drums_out



def objective(trial, nome_brano):
    # Suggerisce i parametri da testare'
    rho = trial.suggest_float("rho", 0.5, 1.2)
    a = trial.suggest_float("leak_rate", 0.1, 1.0)
    input_scale = trial.suggest_float("input_scale", 0.01, 1.0)
    ridge_alpha = trial.suggest_loguniform("alpha", 1e-6, 1.0)
    sparsity = trial.suggest_float("sparsity", 0.01, 1.0)

    train_segments = {k: v for k, v in file_segments.items() if k != nome_brano}
    
    W_in, W_res = init_weights(n_units, n_freq, rho, a, input_scale, seed=42, sparsity=sparsity)
    W_out = train_model(train_segments, W_in, W_res, n_units, a, ridge_alpha)
    drums_pred = separa(f"{nome_brano}_mix.wav", W_in, W_res, W_out, a, directory="Dataset")

    _, drums_target = wavfile.read(f"Dataset/{nome_brano}_drum.wav")
    if len(drums_target.shape) > 1: drums_target = np.mean(drums_target, axis=1)
    drums_target = drums_target.astype(np.float32)
    drums_target /= np.max(np.abs(drums_target)) + 1e-9
    
    # Calcola l'errore (MSE) tra drums_pred e drums_reali
    min_len = min(len(drums_pred), len(drums_target)) #evitiamo errori di shape se le durate differiscono leggermente
    drums_pred = drums_pred[:min_len]
    drums_target = drums_target[:min_len]

    error = np.mean((drums_pred - drums_target)**2)
    return error



def visualizza_risultato(mix_audio, drums_audio, fs, titolo="Confronto Separazione"):
    plt.figure(figsize=(14, 6))

    # Spettrogramma del Mix
    plt.subplot(1, 2, 1)
    D_mix = librosa.amplitude_to_db(np.abs(librosa.stft(mix_audio)), ref=np.max)
    librosa.display.specshow(D_mix, y_axis='log', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mix Originale (Log Scale)')

    # Spettrogramma dei Drums Estratti
    plt.subplot(1, 2, 2)
    D_drums = librosa.amplitude_to_db(np.abs(librosa.stft(drums_audio)), ref=np.max)
    librosa.display.specshow(D_drums, y_axis='log', x_axis='time', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Drums Estratti (ESN)')

    plt.tight_layout()
    plt.show()
  

# --- 3. MAIN EXECUTION FLOW ---

if __name__ == "__main__":
    
    if TRAIN_MODE:
        # 1. Inizializza pesi nuovi
        W_in, W_res = init_weights(n_units, n_freq, rho, a, input_scale)
        
        # 2. Fai il training
        W_out = train_model(file_segments, W_in, W_res, n_units, a, Ridge_alpha)
        
        # 3. Salva tutto il "cervello"
        print(f"Salvataggio modello in {FILENAME_MODELLO}...")
        np.savez(FILENAME_MODELLO, W_in=W_in, W_res=W_res, W_out=W_out)

    if OPTIMIZE_MODE:
        # Esegui l'ottimizzazione dei parametri con Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, nome_brano=FILENAME_OTTIMIZZAZIONE), n_trials=50)
        print("Migliori parametri trovati:", study.best_params)
        exit()    
        
    else:
        # Carica modello esistente
        if os.path.exists(FILENAME_MODELLO):
            print(f"Caricamento modello da {FILENAME_MODELLO}...")
            data = np.load(FILENAME_MODELLO)
            W_in = data['W_in']
            W_res = data['W_res']
            W_out = data['W_out']
            print("Modello caricato con successo!")
        else:
            print("ERRORE: File modello non trovato. Imposta TRAIN_MODE = True.")
            exit()
    


    # --- TEST DI SEPARAZIONE ---
    # Ora puoi chiamare separa quante volte vuoi
    # Passagli W_in, W_res e W_out esplicitamente
      
    drums_out = separa('soul journey.wav', W_in, W_res, W_out, a, directory="Brani Input")
    # separa('altro_brano_mix.wav', W_in, W_res, W_out, a, directory="Dataset")

    fs, mix_originale = wavfile.read('Brani Input/soul journey.wav')
    if len(mix_originale.shape) > 1: mix_originale = np.mean(mix_originale, axis=1)
    mix_originale = mix_originale.astype(np.float32) / (np.max(np.abs(mix_originale)) + 1e-9)
    visualizza_risultato(mix_originale, drums_out, fs)