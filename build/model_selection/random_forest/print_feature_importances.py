import pickle
from pathlib import Path
import numpy as np
p=Path(r'd:/Coding/Machine learning/YT dislikes/model_selection/random_forest/rf_model.pkl')
D=pickle.load(open(p,'rb'))
model=D['model']; features=D['features']
imp=model.feature_importances_
order=np.argsort(imp)[::-1]
for i in range(min(20,len(features))):
    print(f"{i+1}. {features[order[i]]}: {imp[order[i]]:.6f}")
