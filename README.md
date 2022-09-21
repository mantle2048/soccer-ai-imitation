## Soccer AI imitation with specific goal style (based on GRF 5v5 environment)
### **Tip:**
Execute the following code in workspace to create the folder and place the hdf5 dataset in '. grf/datasets'
```bash
mkdir -p .grf/datasets
```
### **Example:**
Execute the following code for training
```bash
python examples/football_gail.py
or
./examples/football_gail.sh
```
Execute the following code to evaluate a trained model
```bash
python examples/football_evaluate.py
```
### **Framework:**
![framework](figure/framework.png)
