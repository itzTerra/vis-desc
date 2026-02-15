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

export interface Scorer {
  id: string;
  label: string;
  description: string;
  speed: number;
  quality: number;
  disabled: boolean;
  stages: ScorerStage[];
  score(
    data: any,
    onProgress: (progress: ScorerProgress) => void,
    socket?: any
  ): Promise<any[]>;
  dispose(): void;
}
