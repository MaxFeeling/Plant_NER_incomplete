python3 ./BiLstm-partial-ner/train_plant.py\ 
              --dataset data \
              --num_epochs 30 \
              --model_folder saved_model \
              --seed 2020 --device_num cuda \
              --batch_size 20\
              --learning_rate 0.01\
              --optimizer sgd\
              