import torch
import torch.nn as nn
import torch.optim as optim

from data import train_loader, val_loader

#===========================================
# DoubleConv
# Reusable block to repeat encoder code 
# As explained with the help of AI, this code
# does Convolution -> ReLU activation -> 
# Another Convolution -> Another ReLU ->
# Maxpooling (shrinking the image)
#===========================================
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            # in_channels: how many channels come in
            # out_channels: how many feature maps to create
            # kernel_size=3: creates a 3x3 filter
            # padding=1: keeps the image size unchanged (otherwise, it would 
            # shrink after every convolution, and we don't want that)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # First convolution was to learn the simple patterns, this second 
            # one is to learn the more complex patterns (standard practice)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # ReLU adds non-linearity; without this, the CNN would be 
            # just like basic math (all negatives are also ignored)

            # Note: Channels (in this case) are just 'layers 
            # of information'
            # In RGB images, they have 3 channels since they have 
            # 3 colors! Each channel is like a matrix
            # AI said that channels are like 'different ways of looking
            # at the image', and that the deeper we go into the network,
            # the fewer image details we have and the more abstract our understanding is
        )

    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # TODO: define encoder (contracting path)

        # inputs an RGB image with 3 channels, and outputs 64 feature maps
        # Remember how we only had 3 channels that represented the colors?
        # Now, we have a lot more channels to hold information on the details 
        # of the image, like edges, corners, textures, etc!
        self.enc1 = DoubleConv(in_channels, 64)

        # using the 64 channels from before, output 128 channels
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # This shrinks the image dimensions by 2 so that 
        # the strongest features are kept while computation
        # is reduced
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # TODO: define bottleneck (e.g., 512 -> 1024)
        # doubles the amount of channels again, giving us the 
        # most compressed representation of the image
        self.bottleneck = DoubleConv(512, 1024)

        # TODO: define decoder (expanding path)

        # turns 1024 input channels into 512 output channels and doubles spatial size
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
       
        # 1024 input channels b/c 512 unsampled features + 512 skip connection from encoder = 1024 total
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)



        # TODO: define final 1x1 conv (out_channels = num_classes)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        # use a 1x1 convolution because it's commonly used in segmentation 
        # since it does not look at neighboring pixels and it only changes 
        # channels; its job it to reduce the 64 feature channels down into 
        # only 1 segmentation channel


        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: encoder forward, store feature maps for skip connections
        # inputs original image and outputs 64 feature channels
        # and save for later since the decoder will need it
        e1 = self.enc1(x)
        
        # shrink image size and save the version after pooling to p1
        p1 = self.pool(e1)

        # and repeat...
        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)


        # TODO: bottleneck forward
        # this saves the highly compressed image understanding
        b = self.bottleneck(p4)

        # TODO: decoder forward, concat with skip connections
        # now we enlarge image size
        d4 = self.up4(b)
        
        # here, we combine decoder features and encoder features 
        # (the encoder had the detailed spatial info and the decoder 
        # had the semantic understanding)
        # This will give us object understanding and precise boundaries
        # dim=1 means to concatenate the channels, combining 512 and 512 into 1024 channels
        d4 = torch.cat([d4, e4], dim=1)
        
        # This refines the combined features
        # The decoder now learns how to use encoder details while
        # its reconstructing segmentation
        # And of course repeat this 4 times
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1) 
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self. up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # TODO: return segmentation logits via final 1x1 conv
        # This converts 64 channels into one segmentation channel
        # Only segmentation logits are returned, not probabilities yet
        return self.final_conv(d1)

class SegmentationLoss(nn.Module):
    """Loss function for segmentation (e.g., BCE / CrossEntropy / Dice / combined)."""

    # runs one time when creating the loss object
    def __init__(self):
        super().__init__()
        # TODO: define the loss to use

        # This stores the actual loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        #   - BCEWithLogitsLoss for binary segmentation
        #   - CrossEntropyLoss for multi-class segmentation
        #   - optionally combine with Dice loss

    # inputs: logits = the model's predictions
    # targets = the correct answers
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # convert to float because the PyTorch BCE expects floating point tensors,
        # not integers!
        targets = targets.float()

        # This compares every predicted pixel against every correct pixel and then 
        # averages the error
        loss = self.loss_fn(logits, targets)

        # return so that the trainer later can use it to 
        # compute gradients. update weights, and improve predictions
        return loss


class Trainer:
    """Training / validation loop wrapper."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    # Here is where batches are loaded, gradiets are computed, 
    # weights are updated, and learning actually happens

    # simple overview of training:
    # give the image to the model
    # compare prediction to crrect mask 
    # Measure error (loss)
    # Adjust model weights slightly
    # Repeat thousands of times
    def train_one_epoch(self, loader) -> float:
        
        # Just tells PyTorch that we have begun training
        # because some layers act differently during training
        # and evaluation
        # This is standard practice
        self.model.train()

        # We want to average loss across the whole epoch, so we 
        # accumulate batch losses
        total_loss = 0.0

        # TODO: iterate over (images, masks) batches from loader
        #   1) move tensors to device
        #   2) optimizer.zero_grad()
        #   3) forward -> compute loss
        #   4) loss.backward() -> optimizer.step()
        #   5) accumulate and return average loss


        for images, masks in loader:

            # Move data to GPU if available, CPU if not
            # Everything must live on the same device
            images = images.to(self.device)
            masks = masks.to(self.device)

            masks = (masks == 1).float()

            # Clear old gradients because PyTorch accumulates
            # gradients by default
            # Without this line, gradients would stack forever 
            # and training breaks
            # (erase old learning before new learning)
            self.optimizer.zero_grad()

            # Forward pass
            # images -> UNet -> predicted masks
            outputs = self.model(images)

            # Compute loss 
            # compare predicted masks and correct masks, resulting
            # in a single number where lower is better
            loss = self.criterion(outputs, masks)

            # Backpropagation
            # (How should every weight change to reduce the loss?)
            loss.backward()

            # Update weights (applies gradient updates)
            self.optimizer.step()

            # Accumulate loss
            # convert loss into a normal Python number
            total_loss += loss.item()

        # return final average loss
        return total_loss / len(loader)

    # this line tells PyTorch to not track gradients,
    # saving memory, making validation faster, and preventing
    # accidental training
    @torch.no_grad()

    # how well does the model perform on unseen data?
    # basically we're testing for overfitting
    def validate(self, loader) -> float:

        self.model.eval()

        # TODO: run forward only and compute loss / metrics (IoU, Dice, etc.)
        total_loss = 0.0

        for images, masks in loader:

            # Move tensors to device
            images = images.to(self.device)
            masks = masks.to(self.device)

            masks = (masks == 1).float()

            # Forward pass only
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)

            # Accumulate loss
            total_loss += loss.item()

        return total_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: set hyperparameters (lr, num_epochs, num_classes, etc.)
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 1

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, criterion, optimizer, device)

    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # TODO: save best model checkpoint / visualize predictions / report metrics


if __name__ == "__main__":
    main()
