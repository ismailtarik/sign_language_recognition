import argparse


def get_args():
    parser = argparse.ArgumentParser()

    #: Device
    parser.add_argument(
        "--device",
        type=int,
        default=0
    )

    #: Image width
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="width of image frame"
    )

    #: Image height
    parser.add_argument(
        "--height",
        type=int,
        default=960,
        help="height of image frame"
    )

    #: Static image mode
    parser.add_argument(
        '--use_static_image_mode',
        action='store_true'
    )

    #: Number of hands
    parser.add_argument(
        "--max_num_hands",
        type=int,
        default=2,
        help="Number of hands to detect"
    )

    #: Detection confidence
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.7,
        help="Minimum detection confidence"
    )

    #: Tracking confidence
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence"
    )

    #: Bounding Rectangle
    parser.add_argument(
        "--use_brect",
        type=bool,
        default=True,
        help="Control if show bounding rectangle or not"
    )

    #: Record mode
    parser.add_argument(
        "--record_mode",
        type=bool,
        default=False,
        help="Is it in record mode or in Normal mode"
    )


    args = parser.parse_args()
    return args
