we are going to take only 30% of the whole dataset for the training.
	3.	Reduce LSTM complexity - Use 1-2 LSTM layers with 32-64 units instead of deeper architecture
	4.	Minimal hyperparameter tuning - Use defaults, no extensive optimization
	5.	Basic evaluation only - Focus on accuracy, skip extensive metrics analysis'

	create a virtual environment
install requirements.txt
Unzip the `2_training_set.json.gz` 
using `gunzip -c "Data/2_training_set.json.gz" > data/TrainingData.json`

* JSON file is in JSONL format not standard JSON. This means each line is a separate JSON object, not one large JSON array.


 /Users/animesh/IDS_NetML/.venv/bin/python /Users/animesh/IDS_NetML/Preprocessing/label.py