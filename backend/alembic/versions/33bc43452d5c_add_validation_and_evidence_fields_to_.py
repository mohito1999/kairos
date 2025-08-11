"""Add validation and evidence fields to learned_patterns

Revision ID: 33bc43452d5c
Revises: 94b5722156bf
Create Date: 2025-08-11 13:30:40.672264

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '33bc43452d5c'
down_revision: Union[str, Sequence[str], None] = '94b5722156bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Define the Enum type so both functions can use it
pattern_status_enum = sa.Enum(
    'CANDIDATE', 'VALIDATED', 'REJECTED', 'APPROVED', 'ACTIVE', 'PAUSED', 'ARCHIVED', 
    name='pattern_status_enum'
)

def upgrade() -> None:
    """Upgrade schema."""
    # ### The definitive, simplified fix ###

    # STEP 1: Create the new ENUM type that we will use.
    pattern_status_enum.create(op.get_bind(), checkfirst=True)
    
    # STEP 2: Add all the new columns we want, EXCEPT for the status column for now.
    op.add_column('learned_patterns', sa.Column('uplift_score', sa.Float(), nullable=True))
    op.add_column('learned_patterns', sa.Column('p_value', sa.Float(), nullable=True))
    op.add_column('learned_patterns', sa.Column('confidence_interval', sa.JSON(), nullable=True))
    op.add_column('learned_patterns', sa.Column('evidence_payload', sa.JSON(), nullable=True))
    
    # STEP 3: Drop the old, problematic VARCHAR 'status' column entirely.
    op.drop_column('learned_patterns', 'status')

    # STEP 4: Now, add a brand new 'status' column using our new ENUM type.
    op.add_column('learned_patterns', sa.Column('status', pattern_status_enum, nullable=False, server_default='CANDIDATE'))

    # STEP 5: Finally, drop the other old column we no longer need.
    op.drop_column('learned_patterns', 'historical_evidence_count')
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### Perform all operations in the reverse order ###

    # STEP 1: Add back the old 'historical_evidence_count' column.
    op.add_column('learned_patterns', sa.Column('historical_evidence_count', sa.INTEGER(), server_default=sa.text('0'), autoincrement=False, nullable=False))
    
    # STEP 2: Drop the new 'status' column.
    op.drop_column('learned_patterns', 'status')

    # STEP 3: Add back the old VARCHAR 'status' column.
    op.add_column('learned_patterns', sa.Column('status', sa.VARCHAR(), autoincrement=False, nullable=False, server_default=sa.text("'CANDIDATE'::character varying")))

    # STEP 4: Drop the other new columns.
    op.drop_column('learned_patterns', 'evidence_payload')
    op.drop_column('learned_patterns', 'confidence_interval')
    op.drop_column('learned_patterns', 'p_value')
    op.drop_column('learned_patterns', 'uplift_score')

    # STEP 5: Finally, drop the ENUM type from the database.
    pattern_status_enum.drop(op.get_bind(), checkfirst=True)
    # ### end Alembic commands ###