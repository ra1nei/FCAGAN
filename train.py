import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import torch, wandb, os

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    model = create_model(opt)
    model.setup(opt)
    
    # =============== Load pretrained model via folder ===============
    if hasattr(opt, 'pretrained_dir') and opt.pretrained_dir and os.path.isdir(opt.pretrained_dir):
        print(f"🔄 Loading pretrained model from: {opt.pretrained_dir}")
        for name in ['G', 'D']:
            ckpt_path = os.path.join(opt.pretrained_dir, f'latest_net_{name}.pth')
            if os.path.exists(ckpt_path):
                try:
                    net = getattr(model, f'net{name}')
                    state_dict = torch.load(ckpt_path, map_location=model.device)
                    net.load_state_dict(state_dict, strict=False)
                    print(f"✅ Loaded {ckpt_path}")
                except Exception as e:
                    print(f"Could not load {ckpt_path}: {e}")
            else:
                print(f"Missing {ckpt_path}")

    visualizer = Visualizer(opt)
    total_iters = 0

    # =============== WandB setup ===============
    if opt.use_wandb:
        wandb.login()
        wandb.init(
            project=opt.name,
            name=f"run_{time.strftime('FCAGAN_%d%m_%H:%M')}",
            config=vars(opt)
        )

    # =============== Training Loop ===============
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        with tqdm(total=dataset_size, desc=f"Epoch [{epoch}/{opt.n_epochs + opt.n_epochs_decay}]", ncols=100) as pbar:
            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # =============== Forward + Backward ===============
                model.set_input(data)
                model.optimize_parameters()

                # =============== Log metrics ===============
                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                    # ✅ Log numeric losses to wandb
                    if opt.use_wandb:
                        wandb.log(
                            {**losses, "epoch": epoch, "total_iters": total_iters},
                            step=total_iters
                        )

                # =============== Display visuals ===============
                if total_iters % opt.display_freq == 0:
                    save_result = (total_iters % opt.update_html_freq == 0)
                    model.compute_visuals()
                    visuals = model.get_current_visuals()
                    visualizer.display_current_results(visuals, epoch, save_result)

                    # ✅ Log images to wandb (once per display_freq)
                    if opt.use_wandb:
                        images = []
                        for label, image in visuals.items():
                            if torch.is_tensor(image):
                                img = image.detach().cpu()
                                if img.ndim == 3 and img.shape[0] in [1, 3]:
                                    images.append(wandb.Image(img, caption=label))
                        if images:
                            wandb.log({"visuals": images, "epoch": epoch, "total_iters": total_iters}, step=total_iters)

                # =============== Save latest checkpoint ===============
                if total_iters % opt.save_latest_freq == 0:
                    print(f'\nSaving latest model (epoch {epoch}, total_iters {total_iters})')
                    suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                    model.save_networks(suffix)

                iter_data_time = time.time()
                pbar.update(opt.batch_size)

        # =============== End of epoch checkpoint ===============
        if epoch % opt.save_epoch_freq == 0:
            print(f'\nSaving model at end of epoch {epoch} (iters {total_iters})')
            model.save_networks('latest')
            model.save_networks(epoch)

        # =============== End of epoch summary ===============
        epoch_time = time.time() - epoch_start_time
        print(f'End of epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}\tTime Taken: {epoch_time:.1f} sec')
        model.update_learning_rate()
