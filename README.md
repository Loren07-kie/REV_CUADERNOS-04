# MediaLab · Teleconexiones

Proyecto significativo 4 — Lorena Abigail Coca Ccorpa · Ingeniería de Software con IA · SENATI.

Aplicación Flask que integra la gestión de publicidades aceptadas de Teleconexiones con Git/GitHub (semana 10) y visión computacional (semana 11). Rediseño con imágenes originales, fondo con capa oscura, logo generado, tipografía redondeada, animaciones y adaptación a móvil.

## Web publicada

https://teleconexiones-medialab-04.onrender.com

## Funciones

- Campañas: crear, consultar, editar y eliminar, buscar, filtrar por estado, exportar CSV y respaldar/restaurar JSON.
- Cotizador Ridge: diez características, validación de categorías/rangos, precio orientativo y transferencia al formulario de campaña.
- K-Means: tres grupos a partir de duración, emisiones, días y alcance; centroides, silhouette y asignación de campañas.
- Estudio: cámara con permiso, captura, carga de imágenes, escala de grises, desenfoque, bordes, rostros y tintes naranja/violeta con OpenCV real en servidor.
- Galería local IndexedDB, descarga y borrado; no se almacenan imágenes en el servidor.
- Métricas reales del experimento sintético, dataset CSV descargable y cuaderno Jupyter.
- Matriz PEA con evidencias y lista de práctica individual. No se inventan repeticiones o evaluaciones.

## Ejecutar en Windows

Instalar Python 3.11 y abrir `INICIAR.bat`. Crea un entorno, instala dependencias, entrena y abre http://localhost:5000. La primera instalación necesita Internet.

## Ejecutar manualmente

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python entrenamiento/entrenar.py
python app.py
```

## Abrir el cuaderno Jupyter

Con el entorno virtual activado, ejecuta `python -m pip install jupyterlab` y después `python -m jupyterlab entrenamiento/MediaLab.ipynb`. Ejecuta las celdas en orden.

## Validar

```bash
python -m pip install pytest
python -m pytest -q
node --check static/app.js
```

## Render (web service gratuito)

- Repositorio: https://github.com/Loren07-kie/REV_CUADERNOS-04
- Rama: `main`
- Python: 3.11.11
- Build: `pip install -r requirements.txt && python entrenamiento/entrenar.py`
- Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
- Variables: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- Salud: `GET /health` devuelve `status=ok`.
- Un solo proceso limita memoria; OpenCV procesa como máximo 1280×960 tras validar la entrada.

El modelo se genera durante cada build con semilla fija, usando la misma versión de dependencias que la aplicación. No se carga un Joblib subido por visitantes.

## Almacenamiento y límites

Esta entrega es una demostración académica de uso individual. Campañas (hasta 500) y lista de práctica viven en localStorage; galería (hasta 12 fotos) en IndexedDB. No existe base compartida, roles o sincronización entre dispositivos. Un reinicio de Render no borra el almacenamiento del navegador, pero borrar los datos del sitio sí. Descargar respaldo JSON y fotografías antes de cambiar de equipo. JSON no incluye la galería; cada foto se descarga por separado.

Los precios se entrenan con 5 600 datos **sintéticos** generados por una regla con ruido. MAE/R² evalúan esa regla; no acreditan precisión comercial. K-Means identifica grupos exploratorios y no segmentos verificados. No se debe emplear como facturación fiscal ni cotización oficial.

La cámara se apaga al capturar, salir de la sección u ocultar la pestaña. Solo se envía una foto tras consentimiento y acción explícita. Haar detecta posibles rostros, no identifica personas ni diagnostica emociones. Los colores son filtros manuales. No se añadió DeepFace/TensorFlow al servicio gratuito: su ejemplo recibido queda documentado como antecedente, no como una función desplegada.

## Estructura

```
app.py                     API Flask y OpenCV
static/                    Interfaz, estilos, imágenes y JavaScript
entrenamiento/entrenar.py   Generación, validación, Ridge, K-Means
entrenamiento/MediaLab.ipynb Cuaderno explicativo ejecutable
entrenamiento/*.csv         Datos sintéticos y particiones
entrenamiento/resultados/   Métricas del experimento
entrenamiento/modelos.joblib Generado en build; no versionado
tests/                     Pruebas de API, imágenes y separación de datos
docs/                      PEA, manual y validación
```

## Referencias

- OpenCV: https://docs.opencv.org/4.x/
- Scikit-learn: https://scikit-learn.org/stable/
- Flask: https://flask.palletsprojects.com/
- Render: https://render.com/docs/free

Los fondos y el logo son imágenes originales generadas para este proyecto. Fuente Fredoka y DM Sans desde Google Fonts, con alternativas locales si la red no permite cargarlas. El informe de formación práctica se elaborará en una segunda etapa.
