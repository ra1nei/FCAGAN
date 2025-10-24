import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import torch, wandb

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create dataset
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create model
    model.setup(opt)               # load networks and schedulers
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        # use tqdm
        with tqdm(total=dataset_size, desc=f"Epoch [{epoch}/{opt.n_epochs + opt.n_epochs_decay}]", ncols=100) as pbar:
            for i, data in enumerate(dataset):
                iter_start_time = time.time()

                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters()

                # Visualization
                # if total_iters % opt.display_freq == 0:
                #     save_result = total_iters % opt.update_html_freq == 0
                #     model.compute_visuals()
                #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # WanDB
                if opt.use_wandb and total_iters % opt.display_freq == 0:
                    model.compute_visuals()
                    visuals = model.get_current_visuals()
                    images = []
                    for label, image in visuals.items():
                        # convert tensor -> wandb.Image safely
                        if torch.is_tensor(image):
                            image = image.detach().cpu()
                            if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C,H,W)
                                images.append(wandb.Image(image, caption=label))
                    if images:
                        wandb.log({"visuals": images, "epoch": epoch, "total_iters": total_iters})
                        
                # Logging
                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # Save latest model
                if total_iters % opt.save_latest_freq == 0:
                    print(f'\nSaving the latest model (epoch {epoch}, total_iters {total_iters})')
                    save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
                pbar.update(opt.batch_size)  # update the progress bar

        # Save model at epoch end
        if epoch % opt.save_epoch_freq == 0:
            print(f'\nSaving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # Epoch summary
        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}\t Time Taken: {time.time() - epoch_start_time:.1f} sec')
        model.update_learning_rate()
