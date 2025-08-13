"""Refactor learned_patterns for contrastive engine

Revision ID: 5c07aa4c2f29
Revises: b3456032397f
Create Date: 2025-08-13 14:13:33.146182

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5c07aa4c2f29'
down_revision: Union[str, Sequence[str], None] = 'b3456032397f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### Manually adjusted migration to add ONLY the new columns for the Contrastive Engine ###
    op.add_column('learned_patterns', sa.Column('battleground_context', sa.JSON(), nullable=True))
    op.add_column('learned_patterns', sa.Column('positive_examples', sa.JSON(), nullable=True))
    op.add_column('learned_patterns', sa.Column('negative_examples', sa.JSON(), nullable=True))
    # ### end of manual adjustment ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### Manually adjusted migration to remove ONLY the new columns ###
    op.drop_column('learned_patterns', 'negative_examples')
    op.drop_column('learned_patterns', 'positive_examples')
    op.drop_column('learned_patterns', 'battleground_context')
    # ### end of manual adjustment ###