python C:\multilevellabel_NN\addons\create_indexmap.py classification_train_g3 gen3 voc2012_group3_indexmap.txt 2
python C:\multilevellabel_NN\addons\create_mlabel_dataset.py classification_train_g3 gen3 train.csv gen3\base_label_voc2012_group3_indexmap.txt gen3\label_voc2012_group3_indexmap.txt float 1
python C:\multilevellabel_NN\addons\create_mlabel_dataset.py classification_val_g3 gen3 val.csv gen3\base_label_voc2012_group3_indexmap.txt gen3\label_voc2012_group3_indexmap.txt float 1
python C:\multilevellabel_NN\addons\randomize_csv.py gen3\train_train.csv gen3\rtrain_train.csv
python C:\multilevellabel_NN\addons\randomize_csv.py gen3\train_val.csv gen3\rtrain_val.csv