import models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def get_model(args):
    if "resnet" in args.model:
        model = models.__dict__[args.model](num_classes=args.num_classes)

    elif "mlp" in args.model:
        model = models.__dict__[args.model](
            hidden_width=args.width,
            num_classes=args.num_classes,
            activation=args.activation,
            image_size=args.size,
            in_channels=args.in_channels,
        )
    elif "trial" in args.model:
        model = models.__dict__[args.model](num_classes=args.num_classes)

    else:
        print("=> creating model '{}'".format(args.model))
        model = models.__dict__[args.model](
            c=args.width, num_classes=args.num_classes, activation=args.activation
        )

    return model.to(args.device)
