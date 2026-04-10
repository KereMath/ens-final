"""
hopefullyprojectfinal - Ana Script
Stacked Generalization: Eski + Yeni Ensemble → Meta-Learner

Kullanim:
    python main.py                # tam pipeline
    python main.py --force        # cache temizle, bastan isle
    python main.py --eval         # sadece evaluation (meta modeller hazir)
    python main.py --train        # sadece meta-learner egitimi
"""
import sys


def main():
    force      = "--force" in sys.argv
    eval_only  = "--eval"  in sys.argv
    train_only = "--train" in sys.argv

    print("=" * 70)
    print("  hopefullyprojectfinal - Stacked Generalization Ensemble")
    print("  Eski (9 det) + Yeni (10 model) -> 19 meta-feature -> XGBoost")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Adim 1: Meta-learner verisi + egitim
    # ------------------------------------------------------------------
    if not eval_only:
        from processor import prepare_meta_data
        from trainer import train_all_meta

        print("\n>> Adim 1/2: Meta-learner veri hazirlama + egitim")
        meta_X, y_base, y_anom, gids = prepare_meta_data(force=force)
        train_all_meta(meta_X, y_base, y_anom, gids)
    else:
        print("\n>> Adim 1/2: Atlandi (--eval)")

    # ------------------------------------------------------------------
    # Adim 2: Full evaluation (39 grup)
    # ------------------------------------------------------------------
    if not train_only:
        from evaluator import evaluate_stacked, load_meta_models, load_tsfresh_scaler
        from processor import load_new_ensemble, load_old_ensemble

        print("\n>> Adim 2/2: Full stacking v4 evaluation (39 grup)")
        old_models = load_old_ensemble()
        new_models = load_new_ensemble()
        base_meta, anom_metas, blend_params, router = load_meta_models()
        tsfresh_scaler = load_tsfresh_scaler()
        evaluate_stacked(old_models, new_models, base_meta, anom_metas,
                         blend_params, router, tsfresh_scaler)
    else:
        print("\n>> Adim 2/2: Atlandi (--train)")

    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
