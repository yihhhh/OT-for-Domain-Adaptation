wandb login 1a2a4b937223367f6631a531c97f21d4f47b8356

echo "run training ..."
python train_mapping.py --group_id 'ot_layers' --exp_id 'n_layer=1' --config 'ot_1layer'
python train_mapping.py --group_id 'ot_layers' --exp_id 'n_layer=2' --config 'ot_2layer'
python train_mapping.py --group_id 'ot_layers' --exp_id 'n_layer=3' --config 'ot_3layer'
python train_mapping.py --group_id 'ot_layers' --exp_id 'n_layer=4' --config 'ot_4layer'
python train_mapping.py --group_id 'ot_layers' --exp_id 'n_layer=5' --config 'ot_5layer'

echo "run knn ..."
python train_knn.py --record
python train_knn.py --map --record --group_id 'ot_layers' --exp_id 'n_layer=1' --config 'ot_1layer'
python train_knn.py --map --record --group_id 'ot_layers' --exp_id 'n_layer=2' --config 'ot_2layer'
python train_knn.py --map --record --group_id 'ot_layers' --exp_id 'n_layer=3' --config 'ot_3layer'
python train_knn.py --map --record --group_id 'ot_layers' --exp_id 'n_layer=4' --config 'ot_4layer'
python train_knn.py --map --record --group_id 'ot_layers' --exp_id 'n_layer=5' --config 'ot_5layer'