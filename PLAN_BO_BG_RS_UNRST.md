# Plan: extraer Bo, Bg, Rs reales desde los outputs de OPM (Norne)

## Contexto

Hoy, `scripts/extractor.py` calcula `Bo_rb_stb`, `Bg_rb_scf` y
`Rs_scf_stb` interpolando las tablas PVT hardcodeadas de SPE9 en la
presión promedio del reservorio (`FPR`). Eso:

1. Filtra el target en cualquier modelo que use ML (estas tres columnas
   están marcadas como leakage, ya verificado empíricamente: solo `Bg`
   alcanza R² 0.994).
2. Es físicamente incorrecto cuando el dataset corresponde a Norne o a
   Volve, porque las PVT de esos campos son distintas a SPE9.

El usuario quiere reemplazar la interpolación PVT-vs-FPR por una
**extracción fiel desde los outputs del simulador**, aceptando mayor
costo de CPU.

## Reality check sobre qué guarda el simulador

OPM Flow no escribe `Bo` ni `Bg` por celda en el `.UNRST`. Sí guarda:

| Keyword | Está en | Comentario |
|---|---|---|
| `PRESSURE` por celda | `.UNRST` | en BARSA para Norne |
| `RS` por celda | `.UNRST` | porque el deck tiene `DISGAS` |
| `RV` por celda | `.UNRST` | porque el deck tiene `VAPOIL` |
| `SOIL`, `SGAS`, `SWAT` | `.UNRST` | saturaciones por celda |
| `PORV` por celda | `.INIT` | estático |
| `Bo`, `Bg` | — | **no se guardan**; OPM los computa al vuelo desde PVTO/PVTG |

Por lo tanto la ruta fiel real es:

1. Parsear las tablas PVTO/PVTG del deck (no de SPE9).
2. Por cada report step, leer `PRESSURE`, `RS`, `RV`, `SOIL`, `SGAS`
   por celda.
3. Evaluar `Bo_i = Bo(P_i, Rs_i)` y `Bg_i = Bg(P_i, Rv_i)` usando las
   tablas del deck.
4. Promediar al campo ponderando por volumen de fase en condiciones de
   reservorio:

   - `Bo_field = Σ(Bo_i · PORV_i · SOIL_i) / Σ(PORV_i · SOIL_i)`
   - `Bg_field = Σ(Bg_i · PORV_i · SGAS_i) / Σ(PORV_i · SGAS_i)`
   - `Rs_field = Σ(Rs_i · PORV_i · SOIL_i) / Σ(PORV_i · SOIL_i)`

Esto es lo más cercano a "extraer Bo, Bg, Rs del simulador" que se
puede hacer sin tocar el código C++ de OPM. Es **estrictamente más
fiel** que la interpolación en FPR — refleja distribución espacial de
presión y saturaciones, no un promedio escalar.

## Out of scope

- Cambiar SPE9 (sus PVTs ya están bien, son las hardcoded actuales —
  el problema solo aplica a modelos METRIC).
- Backfill del dataset Norne completo de 30 sims. Esta iteración corre
  **una única simulación de Norne** end-to-end como prueba; el batch
  completo se hace después si el resultado es razonable.
- Re-entrenar XGBoost (Bo/Bg/Rs siguen siendo leakage y siguen sin
  usarse para predicción; solo cambia su valor reportado).

## Pasos

### 1. Parser de tablas PVT del deck

Nuevo `scripts/pvt.py` con un parser independiente del modelo:

```python
def parse_pvt_include(path: Path, unit_system: Literal["METRIC", "FIELD"]) -> PvtTable: ...
```

Lee bloques `PVTO`, `PVTG`, `PVDG` (los presentes en el include) y
construye una `PvtTable` con:

```python
@dataclass
class PvtTable:
    # Saturated PVTO: para cada (Rs, Pb) hay un Bo y mu_o
    pvto_rs: np.ndarray            # scf/STB
    pvto_pb_psi: np.ndarray        # psia
    pvto_bo: np.ndarray            # rb/STB
    # Extension subsaturada (opcional): dBo/dP arriba de Pb
    pvto_undersat_dBo_dP: float
    # Wet gas (PVTG): grid Pg × Rv → Bg
    pvtg_pg_psi: np.ndarray
    pvtg_rv_grid: list[np.ndarray] # variable por slice de presión
    pvtg_bg_grid: list[np.ndarray]
    # Dry gas (PVDG): Pg → Bg
    pvdg_pg_psi: np.ndarray
    pvdg_bg: np.ndarray
```

Conversiones a FIELD durante el parse:

- Presión: bar → psi `× 14.5038`
- Rs en PVTO: sm³/sm³ → scf/STB `× 5.6146`
- Bo: rm³/sm³ ≡ rb/STB (sin conversión numérica)
- Bg: rm³/sm³ → rb/scf `× 0.17811`

Métodos:

```python
def bo_cell(self, p_psi, rs_scf_stb) -> np.ndarray:
    """Bo(P, Rs). Interpola en la curva saturada y aplica pendiente
    subsaturada cuando P > Pb(Rs)."""

def bg_cell(self, p_psi, rv_scf_stb) -> np.ndarray:
    """Wet gas: bilineal P × Rv. Para PVDG (Volve): solo P."""
```

Tests rápidos: bo_cell evaluada en valores tabulados devuelve la
fila exacta; chequeo de monotonía respecto a P.

### 2. Reader de `.UNRST` + `.INIT`

Nuevo `scripts/unrst_reader.py`:

```python
def load_initial_porv(init_path: Path) -> np.ndarray: ...

def iter_report_steps(unrst_path: Path):
    """Yields (report_step, pressure_cells, rs_cells, rv_cells, soil_cells, sgas_cells)."""
```

Usa `resdata.resfile.ResdataFile`. Convierte presiones de BARSA →
psi para Norne en este punto (centralizar la conversión).

Maneja el caso `RV` ausente (decks sin `VAPOIL`): retorna `None` y el
agregador usa Bg(P) (PVDG).

### 3. Aggregator: por timestep → escalar

Nuevo `scripts/pvt_aggregate.py`:

```python
def aggregate_field_pvt(
    pvt: PvtTable,
    porv: np.ndarray,
    pressure: np.ndarray,
    rs: np.ndarray,
    rv: np.ndarray | None,
    soil: np.ndarray,
    sgas: np.ndarray,
) -> tuple[float, float, float]:
    """Returns (Bo_field, Bg_field, Rs_field)."""
```

Implementa los promedios ponderados de la sección "Reality check".
Edge cases:

- Si `Σ(PORV · SOIL) == 0` (todas las celdas sin oil, raro pero
  posible en steps tardíos): devolver `np.nan`.
- Igual para gas.
- Bo_cell debe estar acotado al rango tabulado; usar `np.clip` y
  loggear si se está extrapolando.

### 4. Integración con `extractor.py`

`DeckConfig` gana dos campos opcionales:

```python
pvt_table: PvtTable | None = None         # parsed at module import
restart_basename: str = ""                 # nombre del .UNRST (default = summary_basename)
init_basename: str = ""                    # idem .INIT
```

`scripts/decks/norne.py` parsea `INCLUDE/PVT/PVT-WET-GAS.INC` al
importarse y rellena `pvt_table`. SPE9 y Volve quedan con
`pvt_table=None` por ahora (siguen usando la rama existente).

`extract_features` gana un parámetro `sim_dir: Path` (para acceder a
`.UNRST` y `.INIT`). Lógica:

```python
if config.pvt_table is not None:
    bo, bg, rs = compute_pvt_from_unrst(
        config=config,
        sim_dir=sim_dir,
        report_step_count=len(tiempo_dias),
    )
else:
    # Fallback actual: pvt_tables.py + FPR
    bo = bo_from_pressure(fpr_psi, pb_shift)
    ...
```

`compute_pvt_from_unrst` orquesta reader + aggregator y devuelve tres
arrays alineados con el eje temporal del SUMMARY.

### 5. Ajuste del runner

`scripts/runner.py` ya tiene `keep_outputs`. Para esta iteración,
forzar `keep_outputs=True` cuando `config.pvt_table is not None` —
necesitamos el `.UNRST` y `.INIT` antes del cleanup. Después de que
`extract_features` corre, el `finally` borra el directorio igual.

Pasarle también `sim_dir` a `extract_features` (hoy solo pasa
`summary_base`).

### 6. Smoke test: una sola simulación de Norne

```bash
python scripts/generate_dataset.py \
    --model norne \
    --n 1 \
    --workers 1 \
    --seed 42 \
    --out datasets/dataset_norne_one_sim_unrst.csv \
    --log datasets/runs_log_norne_one_sim_unrst.csv
```

Nombre intencionalmente nuevo (`_one_sim_unrst.csv`) — **no pisa**
`dataset_norne.csv` (memoria del proyecto + regla explícita del
usuario). El batch completo de 30 sims con la nueva extracción se
hace en una iteración posterior, separada.

Antes de correr: `docker ps | grep openporousmedia | awk '{print
$1}' | xargs docker stop` (regla del proyecto, orphans persisten en
macOS).

### 7. Verificación de valores

1. **Sanity numérica**: Norne arranca a ~250 bar = 3626 psi. Valores
   esperados:
   - `Rs` inicial ~100 sm³/sm³ ≈ 560 scf/STB.
   - `Bo` inicial ~1.2-1.3 rb/STB.
   - `Bg` inicial pequeño (gas comprimido a alta presión).

2. **Comparación FPR-only vs cell-weighted**: graficar las dos series
   (la actual del extractor y la nueva) en un plot superpuesto.
   Diferencia esperada <5% mientras el campo está cerca de
   homogéneo, mayor cuando `Pr` cae por debajo de `Pb` y aparece gas
   libre desigual.

3. **Consistencia con el .UNSMRY**: `Rs_field` debería seguir la
   forma de `FGORH` (instantaneous GOR del summary) en períodos sin
   gas libre. Si difieren cualitativamente, hay un bug en la
   ponderación.

### 8. Plot de verificación

Script chico `scripts/compare_pvt_methods.py` que:

1. Carga `datasets/dataset_norne.csv` (PVT actual, basada en SPE9 +
   FPR).
2. Carga `datasets/dataset_norne_one_sim_unrst.csv` (nueva, real).
3. Para `sim_id == 1` plotea:
   - `Bo` vs tiempo (dos curvas).
   - `Bg` vs tiempo (dos curvas).
   - `Rs` vs tiempo (dos curvas).
4. Guarda `plots_norne/pvt_method_comparison.png`.

No es parte de la fix definitiva — solo prueba que el cambio hizo lo
que debía.

## Costo estimado

- Código: ~5-6 horas. El parser de PVTG (wet gas) es la parte más
  delicada: tablas anidadas (`/` interno por slice de presión).
- Compute: 1 sim Norne ≈ 40-50 s. La sobrecarga del aggregator por
  step es chica (~46k celdas × ~60 steps; np vectorizado).

## Riesgos

- **PVTG parsing roto**: el formato Eclipse anida slices por presión
  separados por `/`. Si el parser se confunde, `Bg` saldrá NaN o
  inconsistente. Mitigación: tests unitarios contra un fragmento
  hardcoded del propio archivo Norne.
- **Reader devuelve arrays en orden distinto al PORV**: `.UNRST` y
  `.INIT` deben coincidir en el mapping cell → index. Verificar que
  ambos provengan del mismo número total de celdas activas.
- **Mucha CPU extra si el aggregator hace algo ingenuo**: usar
  numpy puro, no loops por celda. Aceptable: ~50ms por timestep.

## Definition of done para esta iteración

1. `scripts/pvt.py`, `scripts/unrst_reader.py`,
   `scripts/pvt_aggregate.py` creados y con tests mínimos.
2. `scripts/extractor.py` despacha entre el camino viejo y el nuevo
   según `config.pvt_table`.
3. `scripts/decks/norne.py` instancia el `PvtTable` parseado del
   include real.
4. `datasets/dataset_norne_one_sim_unrst.csv` existe y tiene valores
   razonables de `Bo`, `Bg`, `Rs` en el rango esperado para Norne.
5. `plots_norne/pvt_method_comparison.png` muestra las dos series
   superpuestas.
6. `datasets/dataset_norne.csv` original **no fue modificado** (regla
   explícita del usuario, validada con `git diff`).
