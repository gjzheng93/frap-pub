import runexp
import testexp
import summary
import argparse
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--memo", type=str, default="default")
    parser.add_argument("--algorithm", type=str, default="TransferDQN")
    parser.add_argument("--num_phase", type=int, default=8)
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--run_round", type=int, default=200)

    parser.add_argument("--done", action="store_true")
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--rotation_input", action="store_true")
    parser.add_argument("--conflict_matrix", action="store_true")

    parser.add_argument("--run_counts", type=int, default=3600)
    parser.add_argument("--test_run_counts", type=int, default=3600)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--min_lr", type=float, default=0.001)
    parser.add_argument("--update_q_bar_every_c_round", type=bool, default=False)
    parser.add_argument("--early_stop_loss", type=str, default="val_loss")
    parser.add_argument("--dropout_rate", type=float, default=0)

    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sumo_gui", action="store_true")
    parser.add_argument("--min_action_time", type=int, default=10)
    parser.add_argument("--workers", type=int, default=7)


    parser.add_argument("--visible_gpu", type=str, default="")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    #memo = "multi_phase/optimal_search_new/new_headway_anon"
    memo = args.memo
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    t1 = time.time()
    runexp.main(args, memo)
    print("****************************** runexp ends (generate, train, test)!! ******************************")
    t2 = time.time()
    f_timing = open(os.path.join("records", memo, "timing.txt"), "a+")
    f_timing.write(str(t2-t1)+'\n')
    f_timing.close()
    summary.main(memo)
    print("****************************** summary_detail ends ******************************")
