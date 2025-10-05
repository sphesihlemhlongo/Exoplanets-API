/*
  # NASA Exoplanet Discovery Platform Database Schema

  ## Overview
  Creates the complete database schema for storing predictions, optimization studies, 
  and analysis sessions for the NASA Exoplanet ML platform.

  ## New Tables

  ### 1. predictions
  Stores individual exoplanet prediction results with feature data and confidence scores.
  - `id` (uuid, primary key) - Unique prediction identifier
  - `features` (jsonb) - Input feature dictionary
  - `prediction` (integer) - Binary classification (0=false positive, 1=exoplanet)
  - `confidence` (float) - Model confidence score (0-1)
  - `probabilities` (jsonb) - Class probability distribution
  - `session_id` (uuid, nullable) - Optional session tracking
  - `created_at` (timestamptz) - Prediction timestamp

  ### 2. optimization_studies
  Stores Optuna hyperparameter optimization study results and configurations.
  - `id` (uuid, primary key) - Unique study identifier
  - `config` (jsonb) - Optimization configuration parameters
  - `results` (jsonb) - Complete optimization results including trials
  - `status` (text) - Study status (running, completed, failed)
  - `created_at` (timestamptz) - Study start timestamp
  - `completed_at` (timestamptz, nullable) - Study completion timestamp

  ### 3. analysis_sessions
  Tracks user analysis sessions for result persistence and history.
  - `id` (uuid, primary key) - Unique session identifier
  - `metadata` (jsonb) - Session metadata and configuration
  - `prediction_count` (integer) - Number of predictions in session
  - `created_at` (timestamptz) - Session creation timestamp
  - `last_accessed_at` (timestamptz) - Last access timestamp

  ### 4. saved_candidates
  Stores bookmarked exoplanet candidates for further investigation.
  - `id` (uuid, primary key) - Unique bookmark identifier
  - `prediction_id` (uuid) - Reference to prediction
  - `features` (jsonb) - Candidate feature data
  - `notes` (text, nullable) - User notes about the candidate
  - `tags` (text[], nullable) - Searchable tags
  - `created_at` (timestamptz) - Bookmark timestamp

  ## Security
  - Enable Row Level Security (RLS) on all tables
  - Create permissive policies for anonymous access (suitable for demo/public platform)
  - In production, replace with authenticated user policies

  ## Indexes
  - Add indexes on commonly queried columns for performance optimization
  - Timestamp indexes for efficient sorting and filtering
  - JSONB indexes for feature data queries

  ## Important Notes
  - All timestamps use UTC timezone
  - JSONB columns enable flexible schema evolution
  - UUIDs provide globally unique identifiers
  - Default values ensure data integrity
*/

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  features jsonb NOT NULL,
  prediction integer NOT NULL,
  confidence float NOT NULL,
  probabilities jsonb NOT NULL,
  session_id uuid,
  created_at timestamptz DEFAULT now()
);

-- Create optimization_studies table
CREATE TABLE IF NOT EXISTS optimization_studies (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  config jsonb NOT NULL,
  results jsonb,
  status text NOT NULL DEFAULT 'running',
  created_at timestamptz DEFAULT now(),
  completed_at timestamptz
);

-- Create analysis_sessions table
CREATE TABLE IF NOT EXISTS analysis_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  metadata jsonb DEFAULT '{}'::jsonb,
  prediction_count integer DEFAULT 0,
  created_at timestamptz DEFAULT now(),
  last_accessed_at timestamptz DEFAULT now()
);

-- Create saved_candidates table
CREATE TABLE IF NOT EXISTS saved_candidates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prediction_id uuid REFERENCES predictions(id) ON DELETE CASCADE,
  features jsonb NOT NULL,
  notes text,
  tags text[],
  created_at timestamptz DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_session_id ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);
CREATE INDEX IF NOT EXISTS idx_optimization_studies_status ON optimization_studies(status);
CREATE INDEX IF NOT EXISTS idx_optimization_studies_created_at ON optimization_studies(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_last_accessed ON analysis_sessions(last_accessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_saved_candidates_prediction_id ON saved_candidates(prediction_id);
CREATE INDEX IF NOT EXISTS idx_saved_candidates_created_at ON saved_candidates(created_at DESC);

-- Enable Row Level Security
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization_studies ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_candidates ENABLE ROW LEVEL SECURITY;

-- Create policies for anonymous access (public demo platform)
-- Note: In production, replace these with authenticated user-specific policies

CREATE POLICY "Allow public read access to predictions"
  ON predictions
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public insert to predictions"
  ON predictions
  FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public read access to optimization studies"
  ON optimization_studies
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public insert to optimization studies"
  ON optimization_studies
  FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public update to optimization studies"
  ON optimization_studies
  FOR UPDATE
  TO anon
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow public read access to analysis sessions"
  ON analysis_sessions
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public insert to analysis sessions"
  ON analysis_sessions
  FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public update to analysis sessions"
  ON analysis_sessions
  FOR UPDATE
  TO anon
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow public read access to saved candidates"
  ON saved_candidates
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public insert to saved candidates"
  ON saved_candidates
  FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public delete to saved candidates"
  ON saved_candidates
  FOR DELETE
  TO anon
  USING (true);
