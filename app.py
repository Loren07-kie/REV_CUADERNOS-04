"""MediaLab: inference and image processing. Campaigns remain browser-local."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import base64, io, json, math
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException
from entrenamiento.entrenar import ROOT, CATEGORIES, RANGES, FEATURES
cv2.setNumThreads(1)
Image.MAX_IMAGE_PIXELS=12000000
app=Flask(__name__,static_folder='static');app.config['MAX_CONTENT_LENGTH']=5*1024*1024
bundle=joblib.load(ROOT/'modelos.joblib')
metrics=json.loads((ROOT/'resultados/metricas.json').read_text())

def validate(data):
 if not isinstance(data,dict):raise ValueError('Se requiere un objeto JSON con los datos de la campaña.')
 out={}
 for k,choices in CATEGORIES.items():
  if data.get(k) not in choices:raise ValueError(f'Valor inválido para {k}.')
  out[k]=data[k]
 for k,(a,b) in RANGES.items():
  value=data.get(k)
  if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not a<=value<=b or value!=int(value):raise ValueError(f'{k} debe ser un entero entre {a} y {b}.')
  out[k]=int(value)
 return out

@app.get('/')
def index():return send_from_directory('static','index.html')
@app.get('/health')
def health():return jsonify(status='ok',model='Ridge + K-Means',vision='OpenCV')
@app.get('/api/modelo')
def model():return jsonify(metrics)
@app.post('/api/cotizar')
def quote():
 d=validate(request.get_json(silent=True));df=pd.DataFrame([d]);price=float(bundle['ridge'].predict(df[FEATURES])[0]);group=int(bundle['kmeans'].predict(bundle['scaler'].transform(df[bundle['cluster_features']]))[0])
 return jsonify(precio=round(price,2),grupo=group,mae=metrics['mae'],origen='Modelo entrenado con datos sintéticos; cotización orientativa, no tarifa oficial.')
@app.post('/api/agrupar')
def cluster():
 data=request.get_json(silent=True)
 if not isinstance(data,dict) or not isinstance(data.get('campanas'),list) or not 1<=len(data['campanas'])<=500:raise ValueError('Envía entre 1 y 500 campañas.')
 rows=[validate(d) for d in data['campanas']];df=pd.DataFrame(rows)
 labels=bundle['kmeans'].predict(bundle['scaler'].transform(df[bundle['cluster_features']])).tolist()
 return jsonify(grupos=labels,centros=metrics['kmeans']['centros'],features=bundle['cluster_features'])
@app.post('/api/imagen')
def process_image():
 uploaded=request.files.get('imagen');mode=request.form.get('filtro','original')
 if uploaded is None:raise ValueError('Selecciona una imagen.')
 if mode not in ['original','gris','desenfoque','bordes','rostros','naranja','violeta']:raise ValueError('Filtro no reconocido.')
 try:
  im=Image.open(uploaded.stream)
  if im.width*im.height>12000000:raise ValueError('La imagen debe tener como máximo 12 megapíxeles.')
  im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1280,960));img=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
 except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as e:raise ValueError('No se pudo leer la imagen. Usa JPG, PNG o WebP.') from e
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);faces=[]
 if mode=='gris':img=gray
 elif mode=='desenfoque':img=cv2.GaussianBlur(img,(25,25),0)
 elif mode=='bordes':img=cv2.Canny(gray,80,160)
 elif mode=='rostros':
  cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
  faces=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
  for x,y,w,h in faces:cv2.rectangle(img,(x,y),(x+w,y+h),(60,220,255),3)
 elif mode in ['naranja','violeta']:
  overlay=np.full_like(img,(0,130,255) if mode=='naranja' else (190,50,120));img=cv2.addWeighted(img,.72,overlay,.28,0)
 ok,result=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,90])
 if not ok:raise ValueError('No se pudo generar la imagen.')
 return jsonify(imagen='data:image/jpeg;base64,'+base64.b64encode(result).decode(),rostros=len(faces),luminosidad=round(float(gray.mean()),1),nitidez=round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),1),ancho=img.shape[1],alto=img.shape[0],filtro=mode)
@app.get('/descargas/<name>')
def downloads(name):
 allowed={'datos_sinteticos.csv':ROOT,'metricas.json':ROOT/'resultados','MediaLab.ipynb':ROOT,'OPERACIONES_PEA.md':Path('docs')}
 if name not in allowed:return jsonify(error='Archivo no disponible.'),404
 return send_from_directory(allowed[name].resolve(),name,as_attachment=True)
@app.errorhandler(ValueError)
def invalid(e):return jsonify(error=str(e)),400
@app.errorhandler(HTTPException)
def http_error(e):return jsonify(error='Archivo demasiado grande. Máximo 5 MB.' if e.code==413 else e.description),e.code
@app.after_request
def headers(resp):
 resp.headers['X-Content-Type-Options']='nosniff';resp.headers['Referrer-Policy']='same-origin';resp.headers['Permissions-Policy']='camera=(self), microphone=()';resp.headers['X-Frame-Options']='SAMEORIGIN'
 if request.path.startswith('/api/'):resp.headers['Cache-Control']='no-store'
 return resp
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=False)
