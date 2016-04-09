"""
Example execution script. The dataset parameter can
be modified to coco/flickr30k/flickr8k
"""
import argparse

from capgen import train

parser = argparse.ArgumentParser()
parser.add_argument("--attn_type",  default="deterministic",
                    help="type of attention mechanism")
parser.add_argument("changes",  nargs="*",
                    help="Changes to default values", default="")


def main(params):
    # see documentation in capgen.py for more details on hyperparams
    _, validerr, _ = train(saveto=params["model"],
                           attn_type=params["attn-type"],
                           reload_=params["reload"],
                           dim_word=params["dim-word"],
                           ctx_dim=params["ctx-dim"],
                           dim=params["dim"],
                           n_layers_att=params["n-layers-att"],
                           n_layers_out=params["n-layers-out"],
                           n_layers_lstm=params["n-layers-lstm"],
                           n_layers_init=params["n-layers-init"],
                           n_words=params["n-words"],
                           lstm_encoder=params["lstm-encoder"],
                           decay_c=params["decay-c"],
                           alpha_c=params["alpha-c"],
                           prev2out=params["prev2out"],
                           ctx2out=params["ctx2out"],
                           lrate=params["learning-rate"],
                           optimizer=params["optimizer"],
                           selector=params["selector"],
                           patience=10,
                           maxlen=100,
                           batch_size=16,
                           valid_batch_size=16,
                           validFreq=1000,
                           dispFreq=100,
                           saveFreq=1000,
                           sampleFreq=100,
                           dataset="flickr8k",
                           use_dropout=params["use-dropout"],
                           use_dropout_lstm=params["use-dropout-lstm"],
                           save_per_epoch=params["save-per-epoch"])
    print "Final cost: {:.2f}".format(validerr.mean())


if __name__ == "__main__":
    # default parameters written in train() in capgen.py
    defaults = {"model": "cv/normal/flickr8k.npz",
                "attn-type": "stochastic",
                "dim-word": 100,
                "ctx-dim": 512,
                "dim": 1000,
                "n-layers-att": 1,
                "n-layers-out": 1,
                "n-layers-lstm": 1,
                "n-layers-init": 1,
                "n-words": 10000,
                "lstm-encoder": False,
                "decay-c": 0.,
                "alpha-c": 0.,
                "prev2out": False,
                "ctx2out": False,
                "learning-rate": 0.01,
                "optimizer": "rmsprop",
                "selector": False,
                "use-dropout": False,
                "use-dropout-lstm": False,
                "save-per-epoch": True,
                "reload": False}
    # get updates from command line
    args = parser.parse_args()
    for change in args.changes:
        defaults.update(eval("dict({})".format(change)))
    main(defaults)

