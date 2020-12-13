python3 ./BERT-partial-ner/train_model.py\ 
              --dataset data \
              --num_epochs 30 \
              --model_folder saved_model \
              --seed 2020 \
              --device_num cuda \
              --batch_size 32\
              --learning_rate 1e-4\
              --optimizer adam\
              