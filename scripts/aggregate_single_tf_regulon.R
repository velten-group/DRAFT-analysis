#!/usr/bin/env Rscript

# Build a cluster-specific, multi-timepoint single-TF regulon from existing
# reproduction outputs (base R only).

args <- commandArgs(trailingOnly = TRUE)

options_list <- list(
  result_root = "result",
  tf = "mondo",
  edge_cluster = NA_integer_,
  times = 0:3,
  topn = 25L,
  rrf_k = 60,
  highlight_targets = c("mamo", "kuz")
)

read_option <- function(name, default = NULL) {
  flag <- paste0("--", name)
  position <- match(flag, args)
  if (is.na(position)) {
    return(default)
  }
  if (position == length(args)) {
    stop("Missing value after ", flag)
  }
  args[[position + 1L]]
}

options_list$result_root <- read_option("result-root", options_list$result_root)
options_list$tf <- read_option("tf", options_list$tf)
edge_cluster_arg <- read_option("edge-cluster")
if (!is.null(edge_cluster_arg)) {
  options_list$edge_cluster <- as.integer(edge_cluster_arg)
}
times_arg <- read_option("times")
if (!is.null(times_arg)) {
  options_list$times <- as.integer(strsplit(times_arg, ",", fixed = TRUE)[[1L]])
}
options_list$topn <- as.integer(read_option("topn", options_list$topn))
options_list$rrf_k <- as.numeric(read_option("rrf-k", options_list$rrf_k))
highlight_targets_arg <- read_option("highlight-targets")
if (!is.null(highlight_targets_arg)) {
  options_list$highlight_targets <- trimws(strsplit(highlight_targets_arg, ",", fixed = TRUE)[[1L]])
  options_list$highlight_targets <- options_list$highlight_targets[nzchar(options_list$highlight_targets)]
}

if (anyNA(options_list$times) || length(options_list$times) == 0L) {
  stop("--times must be a comma-separated list of integer time indices")
}
if (is.na(options_list$topn) || options_list$topn <= 0L) {
  stop("--topn must be a positive integer")
}

result_root <- normalizePath(options_list$result_root, mustWork = TRUE)
edge_dir <- file.path(result_root, "germ", "edge_cluster")
network_dir <- file.path(result_root, "germ", "network")
reproduce_dir <- file.path(result_root, "germ", "reproduce")
output_dir <- file.path(result_root, "germ", "single_tf")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

required_files <- c(
  file.path(edge_dir, "edge_clusters_12k_topk.csv"),
  file.path(edge_dir, "edges_selected_topk.csv")
)
missing_required <- required_files[!file.exists(required_files)]
if (length(missing_required) > 0L) {
  stop("Missing required result file(s): ", paste(missing_required, collapse = ", "))
}

read_csv <- function(path) {
  read.csv(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    quote = '"',
    comment.char = ""
  )
}

read_tsv <- function(path) {
  read.delim(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    quote = "",
    comment.char = ""
  )
}

stage_name <- function(time_idx) {
  known <- c(
    "GSC", "Gb", "Sp-2", "Sp-4", "Sp-8", "Sp-16",
    "1deg Sc", "1deg Sc", "1deg Sc",
    "2deg Sc", "2deg Sc", "2deg Sc", "2deg Sc", "2deg Sc", "2deg Sc"
  )
  answer <- paste0("G", time_idx)
  valid <- time_idx >= 0L & time_idx < length(known)
  answer[valid] <- known[time_idx[valid] + 1L]
  answer
}

edge_clusters <- read_csv(required_files[[1L]])
selected_edges <- read_csv(required_files[[2L]])
tf_key <- tolower(options_list$tf)
tf_cluster_edges <- edge_clusters[tolower(edge_clusters$tf_name) == tf_key, , drop = FALSE]
if (nrow(tf_cluster_edges) == 0L) {
  stop("TF was not found in edge-cluster results: ", options_list$tf)
}

tf_indices <- unique(tf_cluster_edges$tf_idx)
if (length(tf_indices) != 1L) {
  stop("Expected exactly one TF index for ", options_list$tf, "; found ", paste(tf_indices, collapse = ","))
}
tf_idx <- as.integer(tf_indices[[1L]])

if (is.na(options_list$edge_cluster)) {
  overlap <- selected_edges[
    tolower(selected_edges$tf_name) == tf_key &
      selected_edges$cell_cluster %in% options_list$times,
    c("tf_idx", "tg_idx", "cell_cluster"),
    drop = FALSE
  ]
  overlap <- merge(
    overlap,
    tf_cluster_edges[c("tf_idx", "tg_idx", "edge_cluster")],
    by = c("tf_idx", "tg_idx")
  )
  if (nrow(overlap) == 0L) {
    stop("No TF edges overlap the requested time window")
  }
  ec_counts <- sort(table(overlap$edge_cluster), decreasing = TRUE)
  options_list$edge_cluster <- as.integer(names(ec_counts)[[1L]])
}

cluster_edges <- tf_cluster_edges[
  tf_cluster_edges$edge_cluster == options_list$edge_cluster,
  c("tf_idx", "tg_idx", "tf_name", "tg_name", "edge_cluster"),
  drop = FALSE
]
cluster_edges <- cluster_edges[!duplicated(cluster_edges$tg_idx), , drop = FALSE]
if (nrow(cluster_edges) == 0L) {
  stop("No edges found for TF=", options_list$tf, " in EC", options_list$edge_cluster)
}
cluster_edges <- cluster_edges[order(cluster_edges$tg_idx), , drop = FALSE]
candidate_idx <- as.integer(cluster_edges$tg_idx)

# Build an index-to-ID map before reading the saved single-TF lists. The saved
# lists contain FBgn IDs but not graph indices.
tg_id_map <- setNames(rep(NA_character_, length(candidate_idx)), as.character(candidate_idx))
network_files <- Sys.glob(file.path(network_dir, "t[0-9][0-9]_TF_TG.tsv"))
for (network_path in network_files) {
  if (!anyNA(tg_id_map)) {
    break
  }
  network_frame <- read_tsv(network_path)
  hit <- network_frame[
    network_frame$TF_idx == tf_idx & network_frame$TG_idx %in% candidate_idx,
    c("TG_idx", "TG_id", "TG_name"),
    drop = FALSE
  ]
  if (nrow(hit) > 0L) {
    tg_id_map[as.character(hit$TG_idx)] <- as.character(hit$TG_id)
    name_positions <- match(hit$TG_idx, cluster_edges$tg_idx)
    replace_name <- !is.na(name_positions) & nzchar(as.character(hit$TG_name))
    cluster_edges$tg_name[name_positions[replace_name]] <- as.character(hit$TG_name[replace_name])
  }
}

# Aggregate the saved stage rankings, which already passed the single-TF
# expression filter (min_expr_frac <= fraction-positive <= max_expr_frac).
stage_lists <- vector("list", length(options_list$times))
stage_re_frames <- vector("list", length(options_list$times))
source_list_lengths <- integer(length(options_list$times))
names(stage_lists) <- paste0("G", options_list$times)
names(stage_re_frames) <- paste0("G", options_list$times)

extract_fbgn <- function(values) {
  parsed <- sub("^.*[(](FBgn[^)]*)[)].*$", "\\1", as.character(values))
  parsed[!grepl("[(]FBgn[^)]*[)]", as.character(values))] <- NA_character_
  parsed
}

extract_gene_name <- function(values) {
  sub(" [(]FBgn[^)]*[)].*$", "", as.character(values))
}

for (time_pos in seq_along(options_list$times)) {
  time_idx <- options_list$times[[time_pos]]
  stage_path <- file.path(
    output_dir,
    sprintf("regulon_TG_RE_%s_Germline_%d.csv", options_list$tf, time_idx)
  )
  if (!file.exists(stage_path)) {
    stop(
      "Missing the expression-filtered stage regulon required for consensus ranking: ",
      stage_path
    )
  }
  stage_frame <- read_csv(stage_path)
  if (!all(c("TG", "RE") %in% names(stage_frame))) {
    stop("Unexpected columns in stage regulon: ", stage_path)
  }
  ordered_targets <- unique(as.character(stage_frame$TG))
  source_list_lengths[[time_pos]] <- length(ordered_targets)
  target_ids <- extract_fbgn(ordered_targets)
  target_names <- extract_gene_name(ordered_targets)
  target_indices <- as.integer(names(tg_id_map)[match(target_ids, tg_id_map)])
  unresolved <- is.na(target_indices)
  if (any(unresolved)) {
    target_indices[unresolved] <- cluster_edges$tg_idx[
      match(target_names[unresolved], cluster_edges$tg_name)
    ]
  }
  stage_list <- data.frame(
    TG = target_names,
    TG_id = target_ids,
    TG_idx = target_indices,
    filtered_rank = seq_along(ordered_targets),
    stringsAsFactors = FALSE
  )
  stage_list <- stage_list[
    !is.na(stage_list$TG_idx) & stage_list$TG_idx %in% candidate_idx,
    ,
    drop = FALSE
  ]
  stage_lists[[time_pos]] <- stage_list

  stage_frame$TG_id <- extract_fbgn(stage_frame$TG)
  stage_frame$TG_name <- extract_gene_name(stage_frame$TG)
  stage_frame$TG_idx <- as.integer(names(tg_id_map)[match(stage_frame$TG_id, tg_id_map)])
  unresolved_re <- is.na(stage_frame$TG_idx)
  if (any(unresolved_re)) {
    stage_frame$TG_idx[unresolved_re] <- cluster_edges$tg_idx[
      match(stage_frame$TG_name[unresolved_re], cluster_edges$tg_name)
    ]
  }
  stage_frame$time_idx <- time_idx
  stage_frame$stage_label <- stage_name(time_idx)
  stage_re_frames[[time_pos]] <- stage_frame[
    !is.na(stage_frame$TG_idx) & stage_frame$TG_idx %in% candidate_idx,
    ,
    drop = FALSE
  ]
}

filtered_candidate_idx <- sort(unique(unlist(lapply(stage_lists, function(frame) frame$TG_idx))))
if (length(filtered_candidate_idx) == 0L) {
  stop("No expression-filtered stage targets overlap EC", options_list$edge_cluster)
}
candidate_idx <- as.integer(filtered_candidate_idx)
cluster_edges <- cluster_edges[match(candidate_idx, cluster_edges$tg_idx), , drop = FALSE]

filtered_rank_matrix <- matrix(
  NA_real_,
  nrow = length(candidate_idx),
  ncol = length(options_list$times),
  dimnames = list(as.character(candidate_idx), paste0("G", options_list$times))
)
for (time_pos in seq_along(stage_lists)) {
  stage_list <- stage_lists[[time_pos]]
  positions <- match(stage_list$TG_idx, candidate_idx)
  filtered_rank_matrix[positions, time_pos] <- stage_list$filtered_rank
}

# RRF gives no contribution when a target is absent from a saved filtered Top
# list. Geometric rank uses the stage list length + 1 for such right-censored
# targets; a normalized variant corrects for stage lists of different length.
rrf_terms <- 1 / (options_list$rrf_k + filtered_rank_matrix)
rrf_terms[is.na(rrf_terms)] <- 0
rrf_score <- rowMeans(rrf_terms)

missing_rank_matrix <- filtered_rank_matrix
normalized_rank_matrix <- filtered_rank_matrix
for (time_pos in seq_along(options_list$times)) {
  list_length <- source_list_lengths[[time_pos]]
  missing_rank_matrix[is.na(missing_rank_matrix[, time_pos]), time_pos] <- list_length + 1
  normalized_rank_matrix[, time_pos] <- (
    normalized_rank_matrix[, time_pos] - 0.5
  ) / max(1, list_length)
  normalized_rank_matrix[is.na(normalized_rank_matrix[, time_pos]), time_pos] <- 1.0
}
geometric_mean_rank <- exp(rowMeans(log(missing_rank_matrix)))
geometric_mean_normalized_rank <- exp(rowMeans(log(normalized_rank_matrix)))

# Continuous embedding scores are retained for interpretation and tie-breaking,
# but do not determine eligibility or the primary consensus order.
read_embedding <- function(path) {
  if (!file.exists(path)) {
    stop("Missing embedding: ", path)
  }
  first_line <- readLines(path, n = 1L, warn = FALSE)
  n_columns <- length(scan(text = first_line, quiet = TRUE))
  values <- scan(path, quiet = TRUE)
  if (n_columns <= 0L || length(values) %% n_columns != 0L) {
    stop("Cannot infer embedding dimensions for ", path)
  }
  matrix(values, ncol = n_columns, byrow = TRUE)
}

score_matrix <- matrix(
  NA_real_,
  nrow = length(candidate_idx),
  ncol = length(options_list$times),
  dimnames = list(as.character(candidate_idx), paste0("G", options_list$times))
)
for (time_pos in seq_along(options_list$times)) {
  time_idx <- options_list$times[[time_pos]]
  tf_embedding <- read_embedding(file.path(reproduce_dir, sprintf("embedding_germline_%d_TF.txt", time_idx)))
  tg_embedding <- read_embedding(file.path(reproduce_dir, sprintf("embedding_germline_%d_TG.txt", time_idx)))
  tf_vector <- tf_embedding[tf_idx + 1L, ]
  tg_subset <- tg_embedding[candidate_idx + 1L, , drop = FALSE]
  denominator <- sqrt(rowSums(tg_subset * tg_subset)) * sqrt(sum(tf_vector * tf_vector))
  denominator[denominator <= 1e-12] <- NA_real_
  score_matrix[, time_pos] <- as.vector(tg_subset %*% tf_vector) / denominator
  rm(tf_embedding, tg_embedding, tg_subset)
  invisible(gc())
}
mean_cos_sim <- rowMeans(score_matrix)

rank_for_best <- filtered_rank_matrix
rank_for_best[is.na(rank_for_best)] <- Inf
best_time_position <- max.col(-rank_for_best, ties.method = "first")
consensus <- data.frame(
  TG = as.character(cluster_edges$tg_name),
  TG_id = unname(tg_id_map[as.character(candidate_idx)]),
  TG_idx = candidate_idx,
  edge_cluster = as.integer(options_list$edge_cluster),
  rrf_score = as.numeric(rrf_score),
  geometric_mean_rank = as.numeric(geometric_mean_rank),
  geometric_mean_normalized_rank = as.numeric(geometric_mean_normalized_rank),
  mean_cos_sim = as.numeric(mean_cos_sim),
  min_cos_sim = apply(score_matrix, 1L, min),
  max_cos_sim = apply(score_matrix, 1L, max),
  filtered_top_stage_count = rowSums(!is.na(filtered_rank_matrix)),
  best_stage = stage_name(options_list$times[best_time_position]),
  best_filtered_rank = filtered_rank_matrix[cbind(seq_len(nrow(filtered_rank_matrix)), best_time_position)],
  stringsAsFactors = FALSE
)

for (time_pos in seq_along(options_list$times)) {
  token <- paste0("G", options_list$times[[time_pos]])
  consensus[[paste0("score_", token)]] <- score_matrix[, time_pos]
  consensus[[paste0("filtered_rank_", token)]] <- filtered_rank_matrix[, time_pos]
  consensus[[paste0("in_filtered_top_", token)]] <- !is.na(filtered_rank_matrix[, time_pos])
}

consensus <- consensus[
  order(
    -consensus$rrf_score,
    consensus$geometric_mean_normalized_rank,
    -consensus$mean_cos_sim,
    consensus$TG
  ),
  ,
  drop = FALSE
]
consensus$consensus_rank <- seq_len(nrow(consensus))
consensus$geometric_rank <- rank(consensus$geometric_mean_normalized_rank, ties.method = "min")
consensus <- consensus[c(
  "consensus_rank", "geometric_rank", "TG", "TG_id", "TG_idx", "edge_cluster",
  "rrf_score", "geometric_mean_rank", "geometric_mean_normalized_rank",
  "mean_cos_sim", "min_cos_sim", "max_cos_sim",
  "filtered_top_stage_count", "best_stage", "best_filtered_rank",
  setdiff(names(consensus), c(
    "consensus_rank", "geometric_rank", "TG", "TG_id", "TG_idx", "edge_cluster",
    "rrf_score", "geometric_mean_rank", "geometric_mean_normalized_rank",
    "mean_cos_sim", "min_cos_sim", "max_cos_sim",
    "filtered_top_stage_count", "best_stage", "best_filtered_rank"
  ))
)]

# Remove provisional Drosophila gene symbols such as CG7768/CG3124, then
# refill the requested Top N from the next named genes in the same consensus
# order. The pre-filter rank is kept; consensus_rank ranks the exported genes.
unnamed_gene_pattern <- "^CG[0-9]+"
consensus$rank_before_name_filter <- consensus$consensus_rank
unnamed_gene_mask <- grepl(unnamed_gene_pattern, consensus$TG, ignore.case = TRUE)
excluded_unnamed_count <- sum(unnamed_gene_mask)
consensus <- consensus[!unnamed_gene_mask, , drop = FALSE]
consensus$consensus_rank <- seq_len(nrow(consensus))
consensus$geometric_rank <- rank(consensus$geometric_mean_normalized_rank, ties.method = "min")

top_count <- min(options_list$topn, nrow(consensus))
top_targets <- consensus[seq_len(top_count), , drop = FALSE]
time_token <- paste0("G", min(options_list$times), "-G", max(options_list$times))
prefix <- paste0(tolower(options_list$tf), "_", time_token, "_consensus")
top_table_path <- file.path(output_dir, paste0(prefix, "_top", top_count, ".tsv"))
all_table_path <- file.path(output_dir, paste0(prefix, "_all_candidates.tsv"))
genes_only_path <- file.path(output_dir, paste0(prefix, "_top", top_count, "_genes.txt"))
write.table(top_targets, top_table_path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
write.table(consensus, all_table_path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
writeLines(as.character(top_targets$TG), genes_only_path)

all_re <- do.call(rbind, stage_re_frames)
all_re <- all_re[all_re$TG_idx %in% top_targets$TG_idx, , drop = FALSE]
selected_re_rows <- list()
for (target_pos in seq_len(nrow(top_targets))) {
  target <- top_targets[target_pos, , drop = FALSE]
  target_re <- if (nrow(all_re) > 0L) {
    all_re[all_re$TG_idx == target$TG_idx, , drop = FALSE]
  } else {
    data.frame()
  }
  if (nrow(target_re) == 0L) {
    selected_re_rows[[length(selected_re_rows) + 1L]] <- data.frame(
      TF = options_list$tf,
      stage = time_token,
      t_idx = paste(options_list$times, collapse = ","),
      consensus_rank = target$consensus_rank,
      TG = paste0(target$TG, " (", target$TG_id, ")"),
      TG_idx = target$TG_idx,
      RE = NA_character_,
      RE_idx = NA_integer_,
      rank_re = NA_integer_,
      selected_stages = NA_character_,
      selected_stage_count = 0L,
      mean_abs_distance_to_TSS_bp = NA_real_,
      stringsAsFactors = FALSE
    )
    next
  }
  re_groups <- split(target_re, as.character(target_re$RE))
  re_summary <- do.call(rbind, lapply(re_groups, function(group) {
    distances <- suppressWarnings(as.numeric(group$dist_REcenter_to_TSS_bp))
    finite_distance <- distances[is.finite(distances)]
    data.frame(
      RE = as.character(group$RE[[1L]]),
      RE_idx = NA_integer_,
      selected_stages = paste(unique(group$stage_label), collapse = ";"),
      selected_stage_count = length(unique(group$time_idx)),
      occurrence_count = nrow(group),
      mean_abs_distance_to_TSS_bp = if (length(finite_distance) > 0L) mean(abs(finite_distance)) else NA_real_,
      stringsAsFactors = FALSE
    )
  }))
  distance_order <- ifelse(is.na(re_summary$mean_abs_distance_to_TSS_bp), Inf, re_summary$mean_abs_distance_to_TSS_bp)
  re_summary <- re_summary[
    order(-re_summary$selected_stage_count, -re_summary$occurrence_count, distance_order, re_summary$RE),
    ,
    drop = FALSE
  ]
  re_summary <- re_summary[seq_len(min(2L, nrow(re_summary))), , drop = FALSE]
  re_summary$rank_re <- seq_len(nrow(re_summary))
  selected_re_rows[[length(selected_re_rows) + 1L]] <- data.frame(
    TF = options_list$tf,
    stage = time_token,
    t_idx = paste(options_list$times, collapse = ","),
    consensus_rank = target$consensus_rank,
    TG = paste0(target$TG, " (", target$TG_id, ")"),
    TG_idx = target$TG_idx,
    RE = re_summary$RE,
    RE_idx = re_summary$RE_idx,
    rank_re = re_summary$rank_re,
    selected_stages = re_summary$selected_stages,
    selected_stage_count = re_summary$selected_stage_count,
    mean_abs_distance_to_TSS_bp = re_summary$mean_abs_distance_to_TSS_bp,
    stringsAsFactors = FALSE
  )
}

regulon_re <- do.call(rbind, selected_re_rows)
regulon_table_path <- file.path(output_dir, paste0("regulon_TG_RE_", prefix, ".csv"))
write.csv(regulon_re, regulon_table_path, row.names = FALSE, na = "")

draw_regulon <- function() {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  par(mar = c(1, 1, 4, 1), xpd = NA, family = "sans")
  plot.new()
  plot.window(xlim = c(-1.6, 1.6), ylim = c(-1.6, 1.6), asp = 1)
  count <- nrow(top_targets)
  angles <- pi / 2 - seq(0, 2 * pi, length.out = count + 1L)[seq_len(count)]
  tg_x <- 1.18 * cos(angles)
  tg_y <- 1.18 * sin(angles)
  quadrant_colors <- c("#d78fdc", "#f46d6d", "#7db5ff", "#f0a21b")
  tg_colors <- quadrant_colors[((floor(((angles %% (2 * pi)) / (pi / 2))) %% 4L) + 1L)]
  emphasis_mask <- tolower(top_targets$TG) %in% tolower(options_list$highlight_targets)
  tg_colors[emphasis_mask] <- "#FFD54F"

  for (target_pos in seq_len(count)) {
    target_idx <- top_targets$TG_idx[[target_pos]]
    emphasize_target <- isTRUE(emphasis_mask[[target_pos]])
    branch_color <- if (emphasize_target) "#C62828" else adjustcolor("black", alpha.f = 0.42)
    branch_width <- if (emphasize_target) 2.4 else 0.8
    target_re <- regulon_re[regulon_re$TG_idx == target_idx & !is.na(regulon_re$RE), , drop = FALSE]
    re_count <- nrow(target_re)
    if (re_count == 0L) {
      segments(0, 0, tg_x[[target_pos]], tg_y[[target_pos]], col = branch_color, lwd = branch_width)
      next
    }
    offsets <- if (re_count == 1L) 0 else seq(-0.045, 0.045, length.out = re_count)
    for (re_pos in seq_len(re_count)) {
      re_angle <- angles[[target_pos]] + offsets[[re_pos]]
      re_x <- 0.61 * cos(re_angle)
      re_y <- 0.61 * sin(re_angle)
      segments(0, 0, re_x, re_y, col = branch_color, lwd = branch_width)
      segments(re_x, re_y, tg_x[[target_pos]], tg_y[[target_pos]], col = branch_color, lwd = branch_width)
      re_fill <- if (emphasize_target) "#EF5350" else "#e6b0b0"
      points(
        re_x,
        re_y,
        pch = if (re_pos %% 2L == 0L) 23 else 22,
        bg = re_fill,
        col = re_fill,
        cex = if (emphasize_target) 1.45 else 1.0
      )
    }
  }

  points(tg_x, tg_y, pch = 21, bg = tg_colors, col = "white", lwd = 1.2, cex = 2.5)
  if (any(emphasis_mask)) {
    points(
      tg_x[emphasis_mask],
      tg_y[emphasis_mask],
      pch = 21,
      bg = "#FFD54F",
      col = "#C62828",
      lwd = 3.2,
      cex = 3.8
    )
  }
  label_x <- 1.38 * cos(angles)
  label_y <- 1.38 * sin(angles)
  target_labels <- as.character(top_targets$TG)
  target_labels[emphasis_mask] <- toupper(target_labels[emphasis_mask])
  ordinary_mask <- !emphasis_mask
  text(
    label_x[ordinary_mask],
    label_y[ordinary_mask],
    labels = target_labels[ordinary_mask],
    cex = 2.0,
    font = 1,
    col = "#222222"
  )
  if (any(emphasis_mask)) {
    text(
      label_x[emphasis_mask],
      label_y[emphasis_mask],
      labels = target_labels[emphasis_mask],
      cex = 2.25,
      font = 2,
      col = "#B71C1C"
    )
  }

  hex_angles <- seq(0, 2 * pi, length.out = 7L) + pi / 6
  polygon(0.22 * cos(hex_angles), 0.22 * sin(hex_angles), col = NA, border = "#C62828", lwd = 3.0)
  polygon(0.19 * cos(hex_angles), 0.19 * sin(hex_angles), col = "#FFAB7D", border = "#4E342E", lwd = 2.6)
  text(0, 0, labels = toupper(options_list$tf), cex = 2.7, font = 2, col = "#5D170C")
  title(
    main = paste0(options_list$tf, " consensus regulon @ ", time_token, " (EC", options_list$edge_cluster, ", TopTG=", top_count, ")"),
    sub = "RRF of expression-filtered stage Top lists; REs from matching single-TF exports",
    cex.main = 1.35,
    cex.sub = 0.9,
    line = 2
  )
}

png_path <- file.path(output_dir, paste0(prefix, "_top", top_count, "_regulon.png"))
tif_path <- file.path(output_dir, paste0(prefix, "_top", top_count, "_regulon.tif"))
png(png_path, width = 4200, height = 4200, res = 300, bg = "white", pointsize = 12)
draw_regulon()
dev.off()
tiff(tif_path, width = 4200, height = 4200, res = 300, compression = "lzw", bg = "white", pointsize = 12)
draw_regulon()
dev.off()

report_path <- file.path(output_dir, paste0(prefix, "_report.txt"))
report_lines <- c(
  paste0("TF\t", options_list$tf),
  paste0("TF_idx\t", tf_idx),
  paste0("edge_cluster\t", options_list$edge_cluster),
  paste0("time_indices\t", paste(options_list$times, collapse = ",")),
  paste0("stage_labels\t", paste(stage_name(options_list$times), collapse = ",")),
  paste0("candidate_count\t", nrow(consensus)),
  paste0("topn\t", top_count),
  paste0("highlight_targets\t", paste(options_list$highlight_targets, collapse = ",")),
  paste0("unnamed_gene_filter\t", unnamed_gene_pattern),
  paste0("excluded_unnamed_gene_count\t", excluded_unnamed_count),
  "eligibility\tunion of saved stage-specific single-TF Top lists after their original expression-fraction filters, restricted to the selected edge cluster",
  paste0("source_top_list_lengths\t", paste(paste0("G", options_list$times, ":", source_list_lengths), collapse = ",")),
  paste0("primary_ranking\tequal-weight reciprocal-rank fusion of filtered stage ranks (k=", options_list$rrf_k, "; absent list entries contribute 0)"),
  "secondary_ranking\tgeometric mean rank; missing entries use source list length + 1; normalized geometric rank corrects unequal source list lengths",
  "limitation\tranking is right-censored because the result bundle contains filtered Top50 for G0-G2 and Top25 for G3, not every expression-eligible TG",
  paste0("top_table\t", top_table_path),
  paste0("genes_only\t", genes_only_path),
  paste0("all_candidates_table\t", all_table_path),
  paste0("regulon_re_table\t", regulon_table_path),
  paste0("png\t", png_path),
  paste0("tif\t", tif_path)
)
writeLines(report_lines, report_path)

cat("Created consensus regulon outputs:\n")
cat(paste0("- ", c(top_table_path, genes_only_path, regulon_table_path, png_path, tif_path, report_path), collapse = "\n"), "\n")
