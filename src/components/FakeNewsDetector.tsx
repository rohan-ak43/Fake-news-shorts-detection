import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, CheckCircle, Play, Search, Brain, Target, X } from "lucide-react";
import { toast } from "sonner";

interface AnalysisResult {
  prediction: "fake" | "real";
  confidence: number;
  fakeSegments: Array<{
    text: string;
    startTime: number;
    endTime: number;
    reason: string;
  }>;
  originalNews?: string;
  explanation: string;
}

export default function FakeNewsDetector() {
  const [url, setUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [urlError, setUrlError] = useState("");

  // URL validation patterns
  const validUrlPatterns = [
    // YouTube Shorts
    /^https?:\/\/(www\.)?(youtube\.com\/shorts\/|youtu\.be\/)/i,
    // Instagram Reels
    /^https?:\/\/(www\.)?instagram\.com\/reel\//i,
    // Instagram Posts (for short videos)
    /^https?:\/\/(www\.)?instagram\.com\/p\//i,
    // Instagram Stories (temporary content)
    /^https?:\/\/(www\.)?instagram\.com\/stories\//i
  ];

  const validateUrl = (inputUrl: string): boolean => {
    if (!inputUrl.trim()) {
      setUrlError("Please enter a URL");
      return false;
    }

    // Check if it's a valid URL format
    try {
      new URL(inputUrl);
    } catch {
      setUrlError("Please enter a valid URL");
      return false;
    }

    // Check if it matches our supported platforms
    const isValid = validUrlPatterns.some(pattern => pattern.test(inputUrl));
    
    if (!isValid) {
      setUrlError("Only YouTube Shorts and Instagram Reels/Posts are supported");
      return false;
    }

    setUrlError("");
    return true;
  };

  const handleUrlChange = (value: string) => {
    setUrl(value);
    if (value.trim()) {
      validateUrl(value);
    } else {
      setUrlError("");
    }
  };

  const analyzeVideo = async () => {
    if (!validateUrl(url)) {
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);
    setResult(null);

    // Simulate progress
    const progressSteps = [
      { progress: 20, message: "Extracting video captions..." },
      { progress: 40, message: "Processing with AI model..." },
      { progress: 70, message: "Analyzing content authenticity..." },
      { progress: 90, message: "Generating detailed report..." },
      { progress: 100, message: "Analysis complete!" }
    ];

    for (const step of progressSteps) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProgress(step.progress);
      toast.info(step.message);
    }

    // Mock result - in real app, this would come from your Python backend
    const mockResult: AnalysisResult = {
      prediction: "fake",
      confidence: 87.3,
      fakeSegments: [
        {
          text: "Scientists have proven that this miracle cure works 100% of the time",
          startTime: 15,
          endTime: 22,
          reason: "Absolute claims ('100%') are typically unreliable in scientific contexts"
        },
        {
          text: "Big pharma doesn't want you to know this secret",
          startTime: 28,
          endTime: 35,
          reason: "Conspiracy language is a common indicator of misinformation"
        }
      ],
      originalNews: "Recent clinical trials show promising results for the treatment, with 73% efficacy in controlled studies.",
      explanation: "The video contains exaggerated claims and conspiracy theories. Legitimate medical information should cite specific studies and acknowledge limitations."
    };

    setResult(mockResult);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Target className="h-8 w-8 text-primary" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Truth Seeker
            </h1>
          </div>
          <p className="text-muted-foreground text-lg">
            AI-powered fake news detection for YouTube Shorts & Instagram Reels
          </p>
        </div>

        {/* Input Section */}
        <Card className="p-6 border-border/50 bg-card/50 backdrop-blur-sm">
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Input
                    placeholder="Paste YouTube Shorts or Instagram Reel URL here..."
                    value={url}
                    onChange={(e) => handleUrlChange(e.target.value)}
                    className={`bg-background/50 border-border pr-10 ${urlError ? 'border-destructive focus:border-destructive' : ''}`}
                  />
                  {url && (
                    <button
                      onClick={() => {
                        setUrl("");
                        setUrlError("");
                      }}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
                <Button 
                  onClick={analyzeVideo} 
                  disabled={isAnalyzing || !!urlError || !url.trim()}
                  className="bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-all duration-300 shadow-lg disabled:opacity-50"
                >
                  {isAnalyzing ? (
                    <Brain className="h-4 w-4 mr-2 animate-pulse" />
                  ) : (
                    <Search className="h-4 w-4 mr-2" />
                  )}
                  {isAnalyzing ? "Analyzing..." : "Analyze"}
                </Button>
              </div>
              
              {urlError && (
                <div className="flex items-center gap-2 text-sm text-destructive">
                  <AlertTriangle className="h-4 w-4" />
                  {urlError}
                </div>
              )}
              
              <div className="text-xs text-muted-foreground">
                <p className="mb-1">Supported platforms:</p>
                <ul className="space-y-1">
                  <li>• YouTube Shorts (youtube.com/shorts/* or youtu.be/*)</li>
                  <li>• Instagram Reels (instagram.com/reel/*)</li>
                  <li>• Instagram Posts (instagram.com/p/*)</li>
                </ul>
              </div>
            </div>
            
            {isAnalyzing && (
              <div className="space-y-2">
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-muted-foreground text-center">
                  Processing video with AI... {progress}%
                </p>
              </div>
            )}
          </div>
        </Card>

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Overall Result */}
            <Card className={`p-6 border-2 ${
              result.prediction === "fake" 
                ? "border-destructive/50 bg-destructive/5" 
                : "border-success/50 bg-success/5"
            }`}>
              <div className="flex items-center gap-4">
                {result.prediction === "fake" ? (
                  <AlertTriangle className="h-12 w-12 text-destructive" />
                ) : (
                  <CheckCircle className="h-12 w-12 text-success" />
                )}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge 
                      variant={result.prediction === "fake" ? "destructive" : "default"}
                      className={result.prediction === "fake" ? "bg-destructive" : "bg-success"}
                    >
                      {result.prediction.toUpperCase()}
                    </Badge>
                    <span className="text-2xl font-bold">
                      {result.confidence}% Confidence
                    </span>
                  </div>
                  <p className="text-muted-foreground">
                    {result.explanation}
                  </p>
                </div>
              </div>
            </Card>

            {/* Fake Segments */}
            {result.fakeSegments.length > 0 && (
              <Card className="p-6">
                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-destructive" />
                  Suspicious Content Detected
                </h3>
                <div className="space-y-4">
                  {result.fakeSegments.map((segment, index) => (
                    <div key={index} className="border border-destructive/20 rounded-lg p-4 bg-destructive/5">
                      <div className="flex items-center gap-2 mb-2">
                        <Play className="h-4 w-4 text-destructive" />
                        <span className="text-sm text-muted-foreground">
                          {segment.startTime}s - {segment.endTime}s
                        </span>
                      </div>
                      <blockquote className="italic text-foreground mb-2">
                        "{segment.text}"
                      </blockquote>
                      <p className="text-sm text-destructive">
                        <strong>Why this is suspicious:</strong> {segment.reason}
                      </p>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Original News */}
            {result.originalNews && (
              <Card className="p-6 border-success/50 bg-success/5">
                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-success" />
                  Verified Information
                </h3>
                <p className="text-foreground">
                  {result.originalNews}
                </p>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}