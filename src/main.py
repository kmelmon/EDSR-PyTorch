import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    #GetContainerClient(args)
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            print("dir_data is " + args.dir_data)
            #if (args.azureml):
                #args.pre_train = os.path.join(args.dir_data, args.pre_train)
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
