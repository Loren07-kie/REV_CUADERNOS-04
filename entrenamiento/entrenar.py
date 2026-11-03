"""Experimento reproducible: Ridge + K-Means, sin datos personales."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
import joblib

ROOT=Path(__file__).resolve().parent
CATEGORIES={'tipo':['Spot televisivo','Spot radial','Publicidad digital','Mención en vivo','Cobertura de evento'],'canal':['TV','Redes sociales','Multiplataforma','Streaming'],'zona':['Satipo','Río Negro','Pangoa','Mazamari'],'turno':['Mañana','Tarde','Noche']}
RANGES={'duracion_segundos':(10,180),'emisiones':(1,30),'dias_campana':(1,45),'alcance_estimado':(800,90000),'urgente':(0,1),'incluye_edicion':(0,1)}
FEATURES=list(CATEGORIES)+list(RANGES)

def generate(n=5600):
 rng=np.random.default_rng(20261026)
 d={k:rng.choice(v,n) for k,v in CATEGORIES.items()}
 d.update({k:rng.integers(a,b+1,n) for k,(a,b) in RANGES.items()})
 df=pd.DataFrame(d)
 price=120+df.tipo.map(dict(zip(CATEGORIES['tipo'],[140,45,80,30,240])))+df.canal.map(dict(zip(CATEGORIES['canal'],[30,15,90,20])))+df.zona.map(dict(zip(CATEGORIES['zona'],[15,5,20,10])))+df.turno.map({'Mañana':10,'Tarde':20,'Noche':45})+df.duracion_segundos*1.5+df.emisiones*4+df.dias_campana*3+df.alcance_estimado*.001+df.urgente*70+df.incluye_edicion*90+rng.normal(0,9.5,n)
 df['precio_soles']=price.round(2);df.insert(0,'id_sintetico',[f'SIM-{i:05}' for i in range(n)])
 return df

def train():
 ROOT.mkdir(exist_ok=True);(ROOT/'resultados').mkdir(exist_ok=True)
 df=generate();df.to_csv(ROOT/'datos_sinteticos.csv',index=False)
 train_df,test_df=train_test_split(df,test_size=.2,random_state=42)
 train_df.to_csv(ROOT/'entrenamiento.csv',index=False);test_df.to_csv(ROOT/'prueba.csv',index=False)
 prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='error'),list(CATEGORIES)),('num',StandardScaler(),list(RANGES))])
 pipe=Pipeline([('prep',prep),('ridge',Ridge())])
 grid=GridSearchCV(pipe,{'ridge__alpha':[.1,1,5,20]},cv=5,scoring='neg_mean_absolute_error',n_jobs=1)
 grid.fit(train_df[FEATURES],train_df.precio_soles);model=grid.best_estimator_
 y=test_df.precio_soles;p=model.predict(test_df[FEATURES]);dummy=DummyRegressor().fit(train_df[FEATURES],train_df.precio_soles)
 cluster_features=['duracion_segundos','emisiones','dias_campana','alcance_estimado']
 scaler=StandardScaler().fit(train_df[cluster_features]);x=scaler.transform(train_df[cluster_features])
 km=KMeans(n_clusters=3,random_state=42,n_init=10).fit(x)
 centers=scaler.inverse_transform(km.cluster_centers_)
 report={'dataset':'5600 campañas sintéticas; no son tarifas ni contratos reales','semilla':20261026,'entrenamiento':len(train_df),'prueba':len(test_df),'alpha':float(grid.best_params_['ridge__alpha']),'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p)),'mae_referencia':float(mean_absolute_error(y,dummy.predict(test_df[FEATURES]))),'media':float(df.precio_soles.mean()),'mediana':float(df.precio_soles.median()),'varianza':float(df.precio_soles.var()),'desviacion':float(df.precio_soles.std()),'cv':[{'alpha':float(a),'mae':float(-m)} for a,m in zip(grid.cv_results_['param_ridge__alpha'],grid.cv_results_['mean_test_score'])],'kmeans':{'k':3,'silhouette':float(silhouette_score(x,km.labels_,sample_size=1000,random_state=42)),'features':cluster_features,'centros':centers.round(2).tolist(),'cantidades':np.bincount(km.labels_).tolist()},'muestra':[{'real':float(a),'predicho':round(float(b),2)} for a,b in zip(y[:60],p[:60])]}
 joblib.dump({'ridge':model,'kmeans':km,'scaler':scaler,'cluster_features':cluster_features},ROOT/'modelos.joblib')
 (ROOT/'resultados'/'metricas.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({k:report[k] for k in ['mae','rmse','r2','mae_referencia']},indent=2))
 return report
if __name__=='__main__':train()
