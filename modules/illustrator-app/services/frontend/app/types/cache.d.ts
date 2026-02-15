/**
 * Download/cache related types for the cache and scorer system.
 */

export interface Downloadable {
  id: string;
  label: string;
  sizeMb: number;
  download(onProgress: (progress: number) => void): Promise<void>;
}

export interface ModelGroup {
  id: string;
  name: string;
  downloadables: Downloadable[];
}

export interface ScorerStage {
  label: string;
  isMain?: boolean;
}

export interface ScorerProgress {
  stage: string;
  progress: number;
  eta?: number;
}
