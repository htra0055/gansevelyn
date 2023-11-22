import pytorch_lightning as pl
from model import MNISTDataModule, GAN

if __name__ == "__main__":
    # Initialize Lightning Data Module and GAN model
    dm = MNISTDataModule()
    model = GAN()

    callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,  # Save the best model based on validation loss
            mode='min',
        )
    
    early_stop = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,  # Stop training if there is no improvement for 3 epochs
            mode='min',
        )

    # Initialize PyTorch Lightning Trainer with additional options
    trainer = pl.Trainer(
        max_epochs=10,
        #accumulate_grad_batches=4,  # Gradient accumulation for larger effective batch size
        precision=16,  # Use 16-bit precision with automatic mixed-precision (AMP)
        callbacks=[callback, early_stop],
    )

    # Train the GAN model using the provided data module
    trainer.fit(model, dm)

    model.plot_imgs()
