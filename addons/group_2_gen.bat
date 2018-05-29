python C:\multilevellabel_NN\addons\create_indexmap.py classification_train_g2 gen2 voc2012_group2_indexmap.txt 2
python C:\multilevellabel_NN\addons\create_mlabel_dataset.py classification_train_g2 gen2 train.csv gen2\base_label_voc2012_group2_indexmap.txt gen2\label_voc2012_group2_indexmap.txt float 1
python C:\multilevellabel_NN\addons\create_mlabel_dataset.py classification_val_g2 gen2 val.csv gen2\base_label_voc2012_group2_indexmap.txt gen2\label_voc2012_group2_indexmap.txt float 1
python C:\multilevellabel_NN\addons\randomize_csv.py gen2\train_train.csv gen2\rtrain_train.csv
python C:\multilevellabel_NN\addons\randomize_csv.py gen2\train_val.csv gen2\rtrain_val.csv