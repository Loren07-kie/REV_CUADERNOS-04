import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from app import app
from entrenamiento.entrenar import ROOT

@pytest.fixture
def client():return app.test_client()
@pytest.fixture
def campaign():return {'tipo':'Spot televisivo','canal':'TV','zona':'Satipo','turno':'Mañana','duracion_segundos':30,'emisiones':6,'dias_campana':7,'alcance_estimado':15000,'urgente':0,'incluye_edicion':1}
def test_pages_and_assets(client):
 for p in ['/','/health','/static/app.js','/static/style.css','/static/assets/logo.webp','/static/assets/estudio.webp','/descargas/MediaLab.ipynb','/descargas/OPERACIONES_PEA.md']:assert client.get(p).status_code==200,p
 assert client.get('/descargas/app.py').status_code==404

def test_quote_and_cluster(client,campaign):
 r=client.post('/api/cotizar',json=campaign);assert r.status_code==200
 assert 400<r.json['precio']<650
 q=client.post('/api/agrupar',json={'campanas':[campaign,campaign]});assert q.status_code==200;assert len(q.json['grupos'])==2;assert q.json['grupos'][0]==r.json['grupo']

@pytest.mark.parametrize('field,value',[('emisiones',0),('alcance_estimado',90001),('turno','Madrugada'),('duracion_segundos',True),('urgente',.5),('dias_campana','7'),('emisiones',float('nan'))])
def test_reject_invalid(client,campaign,field,value):
 campaign[field]=value;assert client.post('/api/cotizar',json=campaign).status_code==400

def test_no_json(client):
 assert client.post('/api/cotizar',data='x').status_code==400
 assert client.post('/api/agrupar',json={'campanas':[]}).status_code==400

@pytest.mark.parametrize('mode',['original','gris','desenfoque','bordes','rostros','naranja','violeta'])
def test_image_filters(client,mode):
 img=np.zeros((80,120,3),dtype=np.uint8);img[:,60:]=[220,130,30];f=io.BytesIO();Image.fromarray(img).save(f,format='PNG');f.seek(0)
 r=client.post('/api/imagen',data={'imagen':(f,'test.png'),'filtro':mode});assert r.status_code==200,r.data;assert r.json['ancho']==120;assert r.json['alto']==80;assert r.json['imagen'].startswith('data:image/jpeg;base64,');assert 0<=r.json['luminosidad']<=255

def test_bad_image(client):
 assert client.post('/api/imagen',data={'imagen':(io.BytesIO(b'not image'),'test.png')}).status_code==400
 assert client.post('/api/imagen').status_code==400
 assert client.post('/api/imagen',data={'imagen':(io.BytesIO(b'x'*(5*1024*1024+1)),'large.png')}).status_code==413

def test_dataset_and_metrics():
 train=pd.read_csv(ROOT/'entrenamiento.csv');test=pd.read_csv(ROOT/'prueba.csv');assert len(train)==4480 and len(test)==1120;assert not set(train.id_sintetico)&set(test.id_sintetico);assert not train.isna().any().any()
 m=json.loads((ROOT/'resultados/metricas.json').read_text());assert m['mae']<m['mae_referencia'];assert 0<m['r2']<=1;assert sum(m['kmeans']['cantidades'])==4480
