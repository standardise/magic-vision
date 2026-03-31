"""Initial migration - create tables

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create objects table
    op.create_table(
        'objects',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('embedding', Vector(384), nullable=True),
        sa.Column('is_active', sa.Boolean, default=True, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create index for vector similarity search
    op.execute('CREATE INDEX objects_embedding_idx ON objects USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')
    
    # Create object_images table
    op.create_table(
        'object_images',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('object_id', sa.String(36), sa.ForeignKey('objects.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('image_path', sa.String(512), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create counting_sessions table
    op.create_table(
        'counting_sessions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('target_object_ids', sa.ARRAY(sa.String(36)), nullable=False, server_default='{}'),
        sa.Column('class_counts', sa.JSON, default={}),
        sa.Column('total_count', sa.Integer, default=0),
        sa.Column('status', sa.String(50), default='created', index=True),
        sa.Column('camera_source', sa.String(255), nullable=True),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('counting_sessions')
    op.drop_table('object_images')
    op.drop_index('objects_embedding_idx')
    op.drop_table('objects')
    op.execute('DROP EXTENSION IF EXISTS vector')
