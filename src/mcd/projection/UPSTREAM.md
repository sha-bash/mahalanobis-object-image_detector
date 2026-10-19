Source: https://github.com/sha-bash/mahalanobis-concept-drift

Commit: fb3bccd1e4a0768fcfb15817fb4fcb809f2af4ad

projector.py is copied without architectural changes. Copyright and original
license are preserved in LICENSE.upstream. Integration is explicitly requested
by the repository author in this research session; no broader relicensing is implied.

The upstream architecture is Linear–BatchNorm–GELU–Dropout, twice, then Linear,
with dimensions input → 256 → 128 → 64. It is not the previous PCA/ReLU adapter.
The upstream training recipe uses random triplets, margin 1, Adam 1e-3, 20 epochs.
Our instance adaptation uses balanced identity batches, common within-identity
covariance estimated only on training identities, and query prototypes for unseen
identities. Original known-class centroids cannot identify unseen test objects.
