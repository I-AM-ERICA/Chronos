from pathlib import Path


def test_official_trainer_exposes_full_state_resume():
    script = Path("scripts/training/train.py").read_text(encoding="utf-8")
    assert "resume_from_checkpoint: Optional[str] = None" in script
    assert "trainer.train(resume_from_checkpoint=resume_from_checkpoint)" in script
