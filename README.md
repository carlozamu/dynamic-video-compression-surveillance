# Dynamic Video Compression with Motion Detection

Questo repository contiene uno script Python che esegue due passaggi fondamentali per la **compressione dinamica di video** in base al movimento:

1. **Calcolo del flusso ottico** e generazione di:
   - **`overlay.mp4`**: copia del video originale.
   - **`mask.mp4`**: maschera binaria (rettangolarizzata) che evidenzia le aree di movimento.
2. **Compressione finale** con DCT (Discrete Cosine Transform) **basata sul movimento**:
   - Le zone con movimento rimangono a **colori** e subiscono poca compressione.
   - Le zone **senza movimento** vengono compresse in modo più aggressivo e convertite in **bianco e nero**.

---

## Requisiti

- **Python 3.x**  
  Consigliato l’uso di un ambiente virtuale (ad es. con `venv` o `conda`).
- **OpenCV**  
  Per installarlo:
  ```bash
  pip install opencv-python
  ```

---

## Struttura del progetto

```
.
├── main.py                # File principale con la pipeline
├── ../Dataset/input/      # Cartella di input con i file video .mp4
├── ../Dataset/output/     # Cartella di output generata dallo script
└── ...
```

---

## Come funziona

1. **Esecuzione del flusso ottico** (`temporal_smoothing_flow`):
   - Analizza frame per frame per calcolare il **flusso ottico** Farneback su tutto il frame (risoluzione piena).
   - Accumula le aree di movimento in una **coda circolare** di dimensione `window_size`.
   - Applica soglie e morfologia per ottenere una **maschera finale** per ogni frame.
   - Genera due file:
     - **`overlay.mp4`** (copia del video originale).
     - **`mask.mp4`** (video in scala di grigi con le aree di movimento rettangolarizzate).

2. **Compressione basata sul movimento** (`compress_with_motion`):
   - Legge `overlay.mp4` e la `mask.mp4`.
   - Applica la **DCT** (Discrete Cosine Transform) solo sulle zone **statiche** (dove la maschera è nera).
   - Converte tali zone in **bianco e nero**.
   - Crea un file **`compressed.mp4`** con la suddivisione tra zone di movimento (a colori) e zone statiche (in B/N).

---

## Come usare lo script

1. **Organizza la cartella**:
   - Metti i tuoi file `.mp4` da processare in `../Dataset/input/`.
   - Assicurati che `../Dataset/output/` esista o venga creata dallo script.

2. **Esegui lo script**:
   ```bash
   python main.py
   ```
   Dove `main.py` è il nome effettivo del file contenente il codice definitivo.

3. **Verifica i risultati**:
   - Nella cartella `../Dataset/output/`, verrà creata una sottocartella per ogni video di input.
   - Ognuna conterrà:
     - `processing.log` con i log dell’elaborazione.
     - `<nome_video>_overlay.mp4`  
     - `<nome_video>_mask.mp4`  
     - `compressed.mp4`  

---

## Parametri principali

- **`flow_threshold`** (default: 0.5)  
  Soglia di magnitudine del flusso ottico per considerare un pixel in movimento.
- **`alpha_fraction`** (default: 0.2)  
  Percentuale minima di frame “movimentati” all’interno della finestra `window_size` per confermare il movimento.
- **`window_size`** (default: 30)  
  Numero di frame di “storico” da considerare per lo smoothing temporale.
- **`morph_kernel`** (default: 2)  
  Dimensione del kernel per le operazioni morfologiche di apertura/chiusura.
- **`QTY_aggressive`** (nel codice di compressione, default: 100)  
  Valore di quantizzazione per la DCT. Più alto = maggiore compressione nelle zone statiche.

---

## Esempi di utilizzo

- **Impostare un livello di sensibilità maggiore (rileva movimenti più lievi)**:
  ```python
  temporal_smoothing_flow(..., flow_threshold=0.2)
  ```
- **Ridurre la finestra temporale** (meno smoothing, più reattività):
  ```python
  temporal_smoothing_flow(..., window_size=15)
  ```
- **Aumentare la compressione**:
  ```python
  # Nel compress_with_motion
  QTY_aggressive = np.full((8, 8), 120, dtype=np.float32)
  ```

---

## Note aggiuntive

- L’algoritmo funziona meglio su video di medie o alte risoluzioni, ma potrebbe rallentare se i video sono molto lunghi o a risoluzione molto alta.
- Se i log non vengono stampati a schermo, assicurati di aver configurato correttamente i permessi di scrittura su `processing.log`.
- Modifica i parametri (es. `pyr_scale`, `winsize`, ecc.) di `cv2.calcOpticalFlowFarneback` in base alle tue esigenze di **performance** o **accuratezza**.

---

## Licenza

Questo progetto è rilasciato sotto la licenza [MIT](LICENSE) (o altra licenza a tua scelta), permettendo l’uso libero e la modifica del codice.

**Enjoy!**
