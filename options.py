import argparse

class Options:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--data_dir", default='datasets/records_albert', help="path to folder containing training/inference data.")

        ap.add_argument("--output_dir", default="output", help="inference output.")
        ap.add_argument("--output_ext", default="png", choices=["png", "jpg"])

        ap.add_argument("--mode", default="train", choices=["train", "inference"])
        ap.add_argument("--phase", default="coarse+fine", choices=["coarse+fine", "coarse", "fine"], help="choose which generators to train.")

        ap.add_argument("--seed", type=int, help="random seed for training.")
        ap.add_argument("--model_dir", default="saved_models", help="directory to save all of the models.")
        ap.add_argument("--load", default=None, help="specify which saved model to load."
                                                    "By default, loads from the newest checkpoint from latest training session."
                                                    "Put '<ModelSubfolder>[/<CheckpointNumber>]' to load from a specific checkpoint."
                                                    "Put 'False' to start training afresh.")

        ap.add_argument("--image_dim", type=int, default=[1024, 1024], nargs=3, help="width, height of images - input images are resized to this size.")
        ap.add_argument("--steps", type=int, default=100000, help="number of training steps to run.")

        ap.add_argument("--trace", action="store_true", help="turn on full tensorflow tracing - used for debugging.")
        ap.add_argument("--summary_freq", type=int, default=50, help="write (non-image) tensorboard summaries every summary_freq steps.")
        ap.add_argument("--eval_freq", type=int, default=100, help="run an evaluation step and write output images to tensorboard every eval_freq steps.")
        ap.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable.")

        ap.add_argument("--batch_size", type=int, default=1, help="number of images in each training batch.")
        ap.add_argument("--shuffle_size", type=int, default=100000, help="shuffle buffer size for training.")
        ap.add_argument("--use_instmaps", action="store_true", help="enable instance maps.")

        ap.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam.")

        ap.add_argument("--lsgan", help="use lsgan loss instead of the regular ns-gan loss", action="store_true")
        ap.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient.")
        ap.add_argument("--fm_weight", type=float, default=5.0, help="feature matching weight for generator gradient.")

        ap.add_argument("--reflect_padding", type=bool, default=True, help="enable/disable reflection padding. Disabling may reduce VRAM usage.")

        a = ap.parse_args()

        self.opt = a