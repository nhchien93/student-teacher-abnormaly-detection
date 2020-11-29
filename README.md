# student-teacher-abnormaly-detection
Implement Teacher-Student network for abnormaly detection. We used simple model architect for high accuracy.

## How to use

* Download dataset from [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

* Run the predict.py script.
```
cd src
python3 demo.py -img {path to image} -model {path to model}
```

## Result
### Image
![Result](result/demo_sample_all.png)
### Accuracy
For dataset 'grid' (https://www.mvtec.com/company/research/datasets/mvtec-ad/)
![ROC](result/demo_eval.png)

## Working folder structure
├── data   
│   ├── grid    
│   └── carpet  
├── resnet18  
│   ├── student_resnet18.pth    
│   └── teacher_resnet18.pth    
├── trained-model   
├── README.md 
├── requirements.txt 
├── result       
└── src  
    ├── config.py  
    ├── demo.py  
    ├── evaluate.py  
    ├── model.py  
    ├── process_data.py   
    ├── train.py  
    └── utils.py  

## References

### Dataset