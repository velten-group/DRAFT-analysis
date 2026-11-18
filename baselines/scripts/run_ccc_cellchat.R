# Use unmodified, commit-pinned upstream CellChat inference functions.
# A shared single-gene fly LR table is the database. It has no curated complex
# or cofactor fields; these are left empty, not borrowed from another species.
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else 'result/ccc_regulatory_benchmark/20261016'
.libPaths(c(normalizePath('.cache/ccc_cellchat/R-library'), .libPaths()))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(jsonlite))
future::plan(future::sequential)
cfg <- fromJSON(file.path(out,'preparation_manifest.json'))$config
upstream <- file.path(cfg$cellchat_source,'R')
for (filename in c('CellChat_class.R', 'utilities.R', 'database.R', 'modeling.R')) source(file.path(upstream, filename))
expr <- as(readMM(file.path(out, 'expression_genes_by_cells.mtx')), 'CsparseMatrix')
genes <- readLines(file.path(out, 'genes.tsv'))
meta <- read.delim(file.path(out, 'cells.tsv'), check.names=FALSE, stringsAsFactors=FALSE)
rownames(expr) <- genes
colnames(expr) <- meta$cell_id
rownames(meta) <- meta$cell_id
meta$samples <- factor(meta$sample)
db <- read.delim(file.path(out, 'cellchat_fly_interactions.tsv'), check.names=FALSE, stringsAsFactors=FALSE, na.strings=NULL)
rownames(db) <- db$interaction_name
object <- createCellChat(expr, meta=meta, group.by='cell_type')
object@DB <- list(interaction=db,
    complex=data.frame(subunit_1=character(),subunit_2=character(),row.names=character()),
    cofactor=data.frame(cofactor1=character(),cofactor2=character(),row.names=character()),
    geneInfo=data.frame(Symbol=genes))
object <- subsetData(object)
object <- identifyOverExpressedGenes(object, do.fast=FALSE)
features <- unique(object@var.features$features)
write.table(object@var.features$features.info, file.path(out,'cellchat_overexpressed_genes.tsv'), sep='\t', quote=FALSE, row.names=FALSE)
# For a DB containing only single-gene pairs, this is the variable.both=TRUE
# selection rule from identifyOverExpressedInteractions, without its complex
# enumeration (which assumes that the DB contains at least one complex).
db$de_selected <- db$ligand %in% features & db$receptor %in% features
write.table(db, file.path(out,'cellchat_candidate_selection.tsv'), sep='\t', quote=FALSE, row.names=FALSE)
object@LR$LRsig <- db
settings <- list(triMean=list(type='triMean',trim=0.1),truncatedMean=list(type='truncatedMean',trim=0.1),arithmeticMean=list(type='truncatedMean',trim=0))
for (label in cfg$cellchat_means) {
    opt <- settings[[label]]
    cat('\nRunning official computeCommunProb:', label, '\n')
    result <- computeCommunProb(object, type=opt$type, trim=opt$trim, LR.use=db,
        raw.use=TRUE, population.size=FALSE, nboot=cfg$nboot, seed.use=as.integer(cfg$seed), Kh=0.5,n=1)
    mean.fun <- if (label == 'triMean') triMean else function(x) mean(x,trim=opt$trim)
    group.means <- aggregate(t(as.matrix(object@data.signaling)/max(object@data.signaling)),list(object@idents),FUN=mean.fun)
    write.table(group.means,file.path(out,paste0('cellchat_',label,'_scaled_group_means.tsv')),sep='\t',quote=FALSE,row.names=FALSE)
    grid <- expand.grid(sender=dimnames(result@net$prob)[[1]], receiver=dimnames(result@net$prob)[[2]],interaction_name=dimnames(result@net$prob)[[3]],stringsAsFactors=FALSE)
    grid$prob <- as.vector(result@net$prob)
    grid$pval <- as.vector(result@net$pval)
    grid$de_selected <- db$de_selected[match(grid$interaction_name,db$interaction_name)]
    grid$significant <- grid$pval < 0.05 & grid$de_selected
    write.table(grid,file.path(out,paste0('cellchat_',label,'_all_edges.tsv')),sep='\t',quote=FALSE,row.names=FALSE)
    saveRDS(result,file.path(out,paste0('cellchat_',label,'.rds')))
}
capture.output(sessionInfo(),file=file.path(out,'R_sessionInfo.txt'))
manifest <- list(mode='Official upstream functions sourced verbatim; not a full CellChat package installation',
    upstream_commit=readLines(file.path(cfg$cellchat_source,'source_commit.txt')),
    candidate_selection='shared all-pair scores plus separately recorded both-gene DE selection',
    custom_database='single-gene fly pairs from original DRAFT candidate prior; complex/cofactor annotations unavailable',
    inference_cells=ncol(object@data), signaling_genes=nrow(object@data.signaling), candidates=nrow(db),de_selected=sum(db$de_selected),
    nboot=cfg$nboot,seed=cfg$seed,Kh=0.5,n=1,population.size=FALSE,
    package_installation=FALSE,official_computeCommunProb_executed=TRUE)
write_json(manifest,file.path(out,'cellchat_execution_manifest.json'),pretty=TRUE,auto_unbox=TRUE)
