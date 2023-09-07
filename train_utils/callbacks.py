from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks.resswap import ResSwap


def get_callback(args):
    callbacks = []
    if "checkpoint" in args.callbacks:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/acc",
            save_top_k=args.save_top_k,
            auto_insert_metric_name=False,
            save_last=args.save_last,
            filename=args.filename,
            save_on_train_epoch_end=True,
            dirpath=f"{args.dirpath}/{args.experiment_name}",
            verbose=True,
            mode="max",
        )
        callbacks.append(checkpoint_callback)
        
    resswap_callback = ResSwap()
    callbacks.append(resswap_callback)
        

    return callbacks
