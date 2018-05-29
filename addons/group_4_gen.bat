python C:\multilevellabel_NN\addons\create_indexmap.py classification_train_g4 gen4 voc2012_group4_indexmap.txt 2
python C:\multilevellabel_NN\addons\create_mlabel_dataset.py classification_train_g4 gen4 train.csv gen4\base_label_voc2012_group4_indexmap.txt gen4\label_voc2012_group4_indexmap.txt float 1
python C:\multilevellabel_NN\addons\create_mlabel_dataset.py classification_val_g4 gen4 val.csv gen4\base_label_voc2012_group4_indexmap.txt gen4\label_voc2012_group4_indexmap.txt float 1
python C:\multilevellabel_NN\addons\randomize_csv.py gen4\train_train.csv gen4\rtrain_train.csv
python C:\multilevellabel_NN\addons\randomize_csv.py gen4\train_val.csv gen4\rtrain_val.csv