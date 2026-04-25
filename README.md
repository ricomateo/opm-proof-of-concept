# opm-proof-of-concept

Este repositorio es una prueba de concepto de [OPM](https://opm-project.org/) +
pipeline para transformar datos de un simulador de reservorio en un dataset
tabular. 

Contiene scripts para generar un dataset con datos físicos y de
producción de un reservorio de petróleo, a partir de simulaciones realizadas
con OPM Flow sobre el modelo
[SPE9](https://github.com/OPM/opm-data/tree/master/spe9), y un notebook que
entrena un modelo XGBoost para predecir la presión promedio del reservorio.

## SPE9

SPE9 es un modelo público de simulación de reservorios.

Es un modelo "black-oil" con 26 pozos (1 inyector de agua y 25 productores de petróleo) sobre una grilla 3D
de 24×25×15 celdas, con 15 capas de porosidad y permeabilidad heterogéneas.

OPM Flow ejecuta SPE9 en aproximadamente 9 segundos y produce 92 reportes a lo
largo de 900 días simulados. Cada reporte es una foto del campo en ese día con
todas las variables agregadas a nivel reservorio.

## Dataset generado

El dataset generado corresponde a 100 simulaciones del modelo SPE9. Tiene el siguiente esquema

| Columna | Símbolo | Unidad | Categoría | Descripción |
|---|---|---|---|---|
| sim_id | - | - | - | ID de simulación |
| Porosidad | φ | fracción | Estática (Petrofísica) | Fracción de espacio vacío en la matriz rocosa. |
| Permeabilidad_mD | k | mD | Estática (Petrofísica) | Capacidad de la roca para permitir el flujo de fluidos. |
| Espesor_Neto_m | h | m | Estática (Geometría) | Espesor productivo neto de la formación. |
| Area | A | m² | Estática (Geometría) | Área total de roca; multiplicada por h se obtiene el volumen total de roca. |
| Presion_Burbuja_psi | Pb | psi | Termodinámica (PVT) | Presión a la cual el gas en solución comienza a liberarse. |
| Bo_rb_stb | Bo | rb/stb | Termodinámica (PVT) | Factor volumétrico del petróleo (relación fondo-superficie). |
| Bg_rb_scf | Bg | rb/scf | Termodinámica (PVT) | Factor volumétrico del gas (relación fondo-superficie). |
| Rs_scf_stb | Rs | scf/stb | Termodinámica (PVT) | Relación de gas disuelto por barril de petróleo. |
| Caudal_Prod_Petroleo_bbl | qo | bbl/d | Dinámica (Operativa) | Volumen diario de petróleo extraído en superficie. |
| Caudal_Prod_Gas_Mpc | Qg | Mpc/d | Dinámica (Operativa) | Volumen diario de gas producido. |
| Caudal_Iny_Agua_bbl | qwinj | bbl/d | Dinámica (Operativa) | Volumen diario de agua inyectada al reservorio. |
| Prod_Acumulada_Petroleo | Np | bbl | Dinámica (Histórica) | Sumatoria histórica del volumen de petróleo extraído. |
| Prod_Acumulada_Gas | Gp | scf | Dinámica (Histórica) | Sumatoria histórica del volumen de gas extraído. |
| Prod_Acumulada_Agua | Wp | bbl | Dinámica (Histórica) | Sumatoria histórica del volumen de agua extraída. |
| Iny_Acumulada_Agua | Winj | bbl | Dinámica (Histórica) | Sumatoria histórica del volumen de agua inyectada. |
| Presion_Reservorio_psi | Pr | psi | Objetivo (Target) | Presión promedio del reservorio en condiciones de fondo. |


## Variabilidad entre simulaciones

El pipeline ejecuta 100 simulaciones variando los siguientes parámetros del modelo SPE9, para obtener distintos resultados y otorgarle variabilidad al dataset.

| Parámetro | Descripción | Rango |
|---|---|---|
| `qwinj_rate` | caudal máximo del inyector de agua | 2.000 a 9.000 STB/día |
| `qo_rate_high` | caudal máximo de los productores | 750 a 2.250 STB/día |
| `k_mult` | multiplicador global sobre la permeabilidad | 0.5x a 2.0x |
| `phi_mult` | multiplicador global sobre la porosidad | 0.7x a 1.3x |
| `p_init` | presión inicial del reservorio | 3.000 a 4.500 psia |
| `pb_shift` | desplazamiento de la curva de bubble point | -300 a +400 psi |


## Generar el dataset desde cero

### Prerrequisitos

1. Docker
2. Python 3.11+ con dependencias:
   ```bash
   pip install numpy pandas resdata
   ```
3. Docker image de OPM:
   ```bash
   docker pull openporousmedia/opmreleases:latest
   ```

### Ejecución

Desde la raíz del repositorio:

```bash
python3 scripts/generate_dataset.py --n 100 --workers 10 --seed 42 --out dataset.csv
```

Flags relevantes:

- `--n 100` cantidad de simulaciones.
- `--workers 10` workers paralelos (igualar a los CPUs asignados a Docker).
- `--seed 42` semilla para reproducibilidad del LHS.
- `--skip-smoke` saltea los smoke tests.

El tiempo de ejecución depende de en qué máquina se ejecute el script, pero tarda alrededor de 10 minutos.

Genera el archivo `dataset.csv`.


## Descripción del pipeline

El entrypoint del pipeline es el script [`scripts/generate_dataset.py`](scripts/generate_dataset.py), que consta de los siguientes pasos:

### 1. Muestreo (`sampling.py`)

Decide qué 100 combinaciones de los seis parámetros se van a probar.

Usa [Latin Hypercube Sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling),
que cubre el espacio de parámetros uniformemente con pocos puntos: divide
cada eje en N intervalos, toma un valor de cada intervalo y combina los ejes
con un shuffle. Más eficiente que un random uniforme y mucho más barato que
una grilla equidistante.

Devuelve N diccionarios, cada uno con los seis valores para una simulación.

### 2. Templating del deck (`deck_template.py`)

Aplica las variaciones de cada simulación. Recibe un diccionario del paso
anterior y devuelve el texto de un `SPE9.DATA` modificado.

Lee el deck baseline una vez y aplica cinco substituciones, una por
parámetro:

1. **`qwinj_rate`** → caudal del inyector (`WCONINJE`).
2. **`qo_rate_high`** → cap de producción (`WCONPROD`).
3. **`p_init`** → presión inicial del reservorio (`EQUIL`).
4. **`k_mult` y `phi_mult`** → multiplicadores globales sobre permeabilidad y
   porosidad (bloque `MULTIPLY` al final de `GRID`).
5. **`pb_shift`** → desplazamiento de la curva de bubble point (tabla `PVTO`).

Las keywords mencionadas (`WCONINJE`, `WCONPROD`, `EQUIL`, etc) son del
[formato Eclipse](https://opm-project.org/?page_id=955), el lenguaje de
input de OPM Flow.

### 3. Ejecución en paralelo (`runner.py`)

Lanza N (cantidad de workers) simulaciones concurrentes con un `ProcessPoolExecutor`. Cada worker:

1. Escribe el deck modificado y los includes en `runs/sim_NNNN/`.
2. Ejecuta `docker run` sobre la imagen oficial de OPM, con un bind mount
   exclusivo para esa simulación.
3. Parsea el output (`.UNSMRY`) con [resdata](https://github.com/equinor/resdata)
   para construir el DataFrame.
4. Elimina `runs/sim_NNNN/` al terminar (cada simulación deja ~50 MB de
   outputs).

Si OPM falla, el worker persiste el log de error y devuelve un resultado
fallido sin tirar abajo el resto del batch.

### 4. Extracción de features (`extractor.py`)

Lee del `.UNSMRY` los 8 vectores temporales que necesita el schema:

- Presión: `FPR`.
- Caudales: `FOPR`, `FGPR`, `FWIR`.
- Acumulados: `FOPT`, `FGPT`, `FWPT`, `FWIT`.

OPM no exporta `Bo`, `Bg`, `Rs` como promedios del campo, por lo que se
calculan en `pvt_tables.py` interpolando las tablas
[PVT](https://en.wikipedia.org/wiki/PVT_analysis) del deck en función de la
presión de cada timestep.

Las features estáticas (porosidad, permeabilidad, geometría, `Pb`) se
derivan de las constantes del deck + los multiplicadores que recibió cada
simulación.

### 5. Agregación y validación (`generate_dataset.py`)

Concatena los DataFrames de las simulaciones y escribe
`dataset.csv` y `runs_log.csv` con los parámetros y métricas por
simulación. Antes valida:

- ≥ 90% de las simulaciones convergieron.
- Sin NaNs ni columnas acumuladas que decrezcan.
- Bo y Rs en rangos físicos.
- Variabilidad mínima de FPR entre simulaciones (si todas dan la misma
  presión, las variaciones no funcionaron).

Antes del batch principal ejecuta dos *smoke tests*: una simulación con
todos los parámetros en baseline y otra con todos en extremos opuestos. Si
la diferencia de FPR final entre ambas es chica, aborta antes de gastar
tiempo en las 100 simulaciones.


### Visualizaciones

```bash
python scripts/plot_dataset.py
```

Genera 7 PNGs en `plots/`: trayectorias de presión, distribuciones de los
parámetros, sensibilidad de FPR, correlaciones entre features, curvas PVT,
paneles por simulación, acumulados.

## Modelo XGBoost

El notebook `notebooks/xgboost_fpr.ipynb` entrena un XGBoost para predecir
la presión del reservorio a partir de las demás columnas del schema.

La separación entre training y test se hace **a nivel simulación** (80 sims train, 20 test), no a nivel
fila: las filas dentro de una misma simulación están fuertemente
correlacionadas en el tiempo, y un row-shuffle daría métricas optimistas
falsas.

De las 16 columnas del schema, se descartan 5 antes de entrenar:

- **`Bo_rb_stb`, `Bg_rb_scf`, `Rs_scf_stb`**: estas columnas se calculan en
  `pvt_tables.py` interpolando las tablas PVT en función de FPR. Es una
  dependencia funcional determinística del target. Si se incluyen como
  features, el modelo aprende a invertir la lookup PVT y obtiene R² ≈ 1 sin
  haber capturado nada de la física del reservorio.
- **`Espesor_Neto_m` y `Area`**: en SPE9 la geometría de la grilla no se
  modifica entre simulaciones, por lo que estas columnas tienen varianza
  cero. No aportan información y solo ensucian los plots de feature
  importance.

Las 10 features finales son las 3 estáticas restantes (porosidad,
permeabilidad, presión de burbuja), las 3 dinámicas operativas (caudales) y
las 4 acumuladas. Las acumuladas son las que más aportan: encodean el estado
de depleción y le dan "memoria" al modelo no secuencial.

### Ejecución

El notebook se puede ejecutar desde VSCode, instalando la extensión [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
