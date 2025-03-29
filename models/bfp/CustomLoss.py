class CustomLoss(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.1, lambda3=0.05):
        super(CustomLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.similarity_metric = nn.CosineSimilarity(dim=1)

    def forward(self, inputs, buf_inputs, feats, buf_feats, norm_inputs, norm_buf_inputs):
        # Compute similarity loss
        similarity_loss = torch.mean(1 - self.similarity_metric(feats, buf_feats))

        # Compute norm difference loss
        norm_loss = torch.abs(norm_inputs - norm_buf_inputs).mean()

        # Compute variance regularization loss
        variance_loss = torch.var(feats) + torch.var(buf_feats)

        # Total loss
        total_loss = (
            self.lambda1 * similarity_loss +
            self.lambda2 * norm_loss +
            self.lambda3 * variance_loss
        )
        return total_loss
