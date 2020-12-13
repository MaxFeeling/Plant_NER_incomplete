python3 ./BiLstm-partial-ner/train_trans.py\ 
              --dataset data \
              --num_epochs 10 \
              --model_folder saved_model \
              --seed 2020 --device_num cuda \
              --batch_size 20\
              --learning_rate 0.01\
              --optimizer sgd\
              --dataset Plant