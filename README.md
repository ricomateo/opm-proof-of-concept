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

SPE9 es un benchmark público de simulación de reservorios. Es un modelo *black-oil* con
26 pozos (1 inyector de agua y 25 productores de petróleo) sobre una grilla 3D
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
python scripts/generate_dataset.py --n 100 --workers 10 --seed 42 --out dataset.csv
```

Flags relevantes:

- `--n 100` cantidad de simulaciones.
- `--workers 10` workers paralelos (igualar a los CPUs asignados a Docker).
- `--seed 42` semilla para reproducibilidad del LHS.
- `--skip-smoke` saltea los smoke tests.

El tiempo de ejecución depende de en qué máquina se ejecute el script, pero tarda alrededor de 10 minutos.

Genera el archivo `dataset.csv`.
