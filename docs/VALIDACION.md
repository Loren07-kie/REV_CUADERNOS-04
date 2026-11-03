# Validación del proyecto — 2 de noviembre de 2026 (Perú)

## Comprobaciones realizadas

- 19 pruebas automatizadas correctas: rutas, cotización, validación de rangos y tipos, agrupamiento, siete filtros, archivos inválidos/grandes y partición independiente.
- Sintaxis JavaScript comprobada con `node --check static/app.js`.
- Las nueve celdas de código del cuaderno se ejecutaron en orden con Python y un mismo espacio de variables. El kernel Jupyter no pudo iniciarse en el entorno de verificación por restricción de sockets; se validó el contenido de las celdas sin afirmar ejecución en interfaz Jupyter.
- Navegador de escritorio, web publicada: carga de ejemplos, edición, cálculo de cotización, transferencia al formulario y guardado de campaña.
- Cotización predeterminada: S/ 509.77. La campaña creada desde ella fue asignada al grupo 3 de K-Means.
- Carga de una imagen generada de estudio, procesamiento con OpenCV en escala de grises y guardado en galería IndexedDB comprobados en la web. Resultado 1280×720, luminosidad 28.6/255 y varianza de Laplaciano 144.1.
- `/health` respondió HTTP 200 con `status=ok`.
- Revisión visual de escritorio sin solapamientos observados. CSS incluye adaptación móvil, pero no se afirma prueba física en un teléfono.

## Límites de la verificación

No se activó una cámara física en el navegador remoto; el permiso/captura deberá probarse en el dispositivo de la estudiante. Se verificó la ruta alternativa de carga de imágenes. Los respaldos/CSV tienen implementación y validación de formato; no se afirma una prueba completa de restauración en otro dispositivo.

## Resultados del entrenamiento

- 5600 campañas sintéticas; 4480 entrenamiento y 1120 prueba.
- MAE: S/ 7.8119; RMSE: S/ 9.7976; R²: 0.99513.
- Referencia del promedio: MAE S/ 112.2689.
- K-Means, k=3, silhouette 0.16993. Separación limitada; grupos exploratorios.

Los resultados se refieren a una regla sintética y no validan tarifas reales.

![Panel publicado](medialab-validado.jpg)
