
import matplotlib.pyplot as plt
import seaborn as sns
from misc.misc import open_pickle


matrix_s_exact = open_pickle("matrix_s_exact.pkl")
matrix_s_binary = open_pickle("matrix_s_binary.pkl")
matrix_d_exact = open_pickle("matrix_d_exact.pkl")
matrix_d_binary = open_pickle("matrix_d_binary.pkl")
matrix_pbrob_eval = open_pickle("matrix_pbrob_eval.pkl")
matrix_bprob_proposed = open_pickle("matrix_bprob_proposed.pkl")
matrix_bprob_proposed_binary = open_pickle("matrix_bprob_proposed_binary.pkl")



# fig, ax = plt.subplots(4, 2, figsize=(30, 20))
#
# sns.heatmap(matrix_s_binary, ax=ax[0, 0])
# ax[0, 0].set_title("Critical speed (0/1)")
# ax[0, 0].set_xlabel("Freeway segment")
# ax[0, 0].set_ylabel("Time (min)")
# ax[0, 0].invert_yaxis()
#
# sns.heatmap(matrix_s_exact, ax=ax[0, 1])
# ax[0, 1].set_title("Exact speed (km/h)")
# ax[0, 1].set_xlabel("Freeway segment")
# ax[0, 1].set_ylabel("Time (min)")
# ax[0, 1].invert_yaxis()
#
# sns.heatmap(matrix_d_binary, ax=ax[1, 0])
# ax[1, 0].set_title("Critical desnity (0/1)")
# ax[1, 0].set_xlabel("Freeway segment")
# ax[1, 0].set_ylabel("Time (min)")
# ax[1, 0].invert_yaxis()
#
# sns.heatmap(matrix_d_exact, ax=ax[1, 1])
# ax[1, 1].set_title("Exact density (v/km/lane)")
# ax[1, 1].set_xlabel("Freeway segment")
# ax[1, 1].set_ylabel("Time (min)")
# ax[1, 1].invert_yaxis()
#
# sns.heatmap(matrix_pbrob_eval, ax=ax[2, 0])
# ax[2, 0].set_title("Ground truth bottleneck estimations (0/1)")
# ax[2, 0].set_xlabel("Freeway segment")
# ax[2, 0].set_ylabel("Time (min)")
# ax[2, 0].invert_yaxis()
#
# sns.heatmap(matrix_bprob_proposed, ax=ax[2, 1])
# ax[2, 1].set_title("Proposed bottleneck estimations (%)")
# ax[2, 1].set_xlabel("Freeway segment")
# ax[2, 1].set_ylabel("Time (min)")
# ax[2, 1].invert_yaxis()
#
# sns.heatmap(matrix_bprob_proposed_binary, ax=ax[3, 0])
# ax[3, 0].set_title("Proposed bottleneck estimations (0/1)")
# ax[3, 0].set_xlabel("Freeway segment")
# ax[3, 0].set_ylabel("Time (min)")
# ax[3, 0].invert_yaxis()
#
# plt.savefig("results.png")
# # plt.show()

_, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.heatmap(matrix_s_exact, ax=ax, cmap="inferno", cbar_kws={'label': 'Exact speed (km/h)'})
ax.set_xticks(range(0, 161, 10))
ax.set_xticklabels(range(0, 161, 10))
ax.set_xlabel("Freeway segment")
ax.set_yticks(range(0, 25, 2))
ax.set_yticklabels(range(0, 25, 2))
ax.set_ylabel("Time (min)")
ax.invert_yaxis()
# plt.show()
plt.savefig("matrix_s_exact.eps")

_, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.heatmap(matrix_s_binary, ax=ax, cmap="inferno_r", cbar_kws={'label': 'Critical speed (0/1)'})
ax.set_xticks(range(0, 161, 10))
ax.set_xticklabels(range(0, 161, 10))
ax.set_xlabel("Freeway segment")
ax.set_yticks(range(0, 25, 2))
ax.set_yticklabels(range(0, 25, 2))
ax.set_ylabel("Time (min)")
ax.invert_yaxis()
# plt.show()
plt.savefig("matrix_s_binary.eps")

_, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.heatmap(matrix_d_exact, ax=ax, cmap="inferno_r", cbar_kws={'label': 'Exact density (v/km/lane)'})
ax.set_xticks(range(0, 161, 10))
ax.set_xticklabels(range(0, 161, 10))
ax.set_xlabel("Freeway segment")
ax.set_yticks(range(0, 25, 2))
ax.set_yticklabels(range(0, 25, 2))
ax.set_ylabel("Time (min)")
ax.invert_yaxis()
# plt.show()
plt.savefig("matrix_d_exact.eps")

_, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.heatmap(matrix_d_binary, ax=ax, cmap="inferno_r", cbar_kws={'label': 'Critical density (0/1)'})
ax.set_xticks(range(0, 161, 10))
ax.set_xticklabels(range(0, 161, 10))
ax.set_xlabel("Freeway segment")
ax.set_yticks(range(0, 25, 2))
ax.set_yticklabels(range(0, 25, 2))
ax.set_ylabel("Time (min)")
ax.invert_yaxis()
# plt.show()
plt.savefig("matrix_d_binary.eps")

_, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.heatmap(matrix_pbrob_eval, ax=ax, cmap="inferno_r", cbar_kws={'label': 'Ground truth bottleneck estimations (0/1)'})
ax.set_xticks(range(0, 161, 10))
ax.set_xticklabels(range(0, 161, 10))
ax.set_xlabel("Freeway segment")
ax.set_yticks(range(0, 25, 2))
ax.set_yticklabels(range(0, 25, 2))
ax.set_ylabel("Time (min)")
ax.invert_yaxis()
# plt.show()
plt.savefig("matrix_pbrob_eval.eps")

_, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.heatmap(matrix_bprob_proposed, ax=ax, cmap="inferno_r", cbar_kws={'label': 'Proposed bottleneck estimations (%)'})
ax.set_xticks(range(0, 161, 10))
ax.set_xticklabels(range(0, 161, 10))
ax.set_xlabel("Freeway segment")
ax.set_yticks(range(0, 25, 2))
ax.set_yticklabels(range(0, 25, 2))
ax.set_ylabel("Time (min)")
ax.invert_yaxis()
# plt.show()
plt.savefig("matrix_bprob_proposed.eps")

_, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.heatmap(matrix_bprob_proposed_binary, ax=ax, cmap="inferno_r", cbar_kws={'label': 'Proposed bottleneck estimations (0/1)'})
ax.set_xticks(range(0, 161, 10))
ax.set_xticklabels(range(0, 161, 10))
ax.set_xlabel("Freeway segment")
ax.set_yticks(range(0, 25, 2))
ax.set_yticklabels(range(0, 25, 2))
ax.set_ylabel("Time (min)")
ax.invert_yaxis()
# plt.show()
plt.savefig("matrix_bprob_proposed_binary.eps")