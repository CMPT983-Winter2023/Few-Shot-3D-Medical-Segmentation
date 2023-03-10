

from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split

def main():
    with open('./dataset.json', 'r+') as f:
        data=json.load(f)
    print(len(data.get('training')))

    train_data,val_data=train_test_split(data['training'], test_size=0.1,random_state=42)

    data['training']=train_data
    data['validation']=val_data

    with open('./dataset.json','w') as f:
        json.dump(data,f)
    

    print(len(data.get('training')))
    print(len(data['validation']))

    


if __name__ == "__main__":
    main()

