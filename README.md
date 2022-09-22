## Soccer AI imitation with specific goal style (based on GRF 5v5 environment)
### **Tip:**
Run the following code to clone the project and setup
```bash
git clone https://github.com/mantle2048/soccer-ai-imitation
cd soccer-ai-imitation
pip install -e .
```
Run the following code in current directory to create the folder and place the hdf5 dataset in `.grf/datasets`
```bash
mkdir -p .grf/datasets
```
### **Example:**
Run the following code for training
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
