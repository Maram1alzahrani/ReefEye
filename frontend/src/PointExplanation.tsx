import { useEffect, useId, useRef, useState } from 'react'
import './PointExplanation.css'

type Props = {
  apiBase: string
  file: Blob
  filename: string
  point: {
    point: number
    row: number
    column: number
    prediction: string
  }
  disabled?: boolean
}

type Explanation = {
  method: string
  row: number
  column: number
  target_label: string
  target_confidence: number
  prediction: string
  has_signal: boolean
  crop_image: string
  heatmap_image: string | null
}

const pretty = (value: string) =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, letter => letter.toUpperCase())

function validExplanation(
  value: unknown
): value is Explanation {
  if (!value || typeof value !== 'object') return false

  const data = value as Explanation

  return (
    data.method === 'Grad-CAM' &&
    Number.isInteger(data.row) &&
    Number.isInteger(data.column) &&
    typeof data.target_label === 'string' &&
    typeof data.prediction === 'string' &&
    Number.isFinite(data.target_confidence) &&
    data.target_confidence >= 0 &&
    data.target_confidence <= 1 &&
    typeof data.has_signal === 'boolean' &&
    typeof data.crop_image === 'string' &&
    data.crop_image.startsWith('data:image/png;base64,') &&
    (
      data.has_signal
        ? typeof data.heatmap_image === 'string' &&
          data.heatmap_image.startsWith(
            'data:image/png;base64,'
          )
        : data.heatmap_image === null
    )
  )
}

export default function PointExplanation({
  apiBase,
  file,
  filename,
  point,
  disabled = false,
}: Props) {
  const [result, setResult] =
    useState<Explanation | null>(null)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [opacity, setOpacity] = useState(0.65)

  const controllerRef =
    useRef<AbortController | null>(null)

  const sliderId = useId()

  useEffect(() => {
    setResult(null)
    setError('')
    setBusy(false)
    setShowHeatmap(true)

    return () => {
      controllerRef.current?.abort()
      controllerRef.current = null
    }
  }, [
    apiBase,
    file,
    point.row,
    point.column,
    point.prediction,
  ])

  async function explain() {
    if (busy || disabled) return

    const controller = new AbortController()
    controllerRef.current = controller

    let timedOut = false

    const timer = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, 120_000)

    setBusy(true)
    setError('')

    try {
      const body = new FormData()

      body.append('file', file, filename)
      body.append('row', String(point.row))
      body.append('column', String(point.column))
      body.append('target_label', point.prediction)

      const response = await fetch(
        `${apiBase.replace(/\/$/, '')}/explain`,
        {
          method: 'POST',
          body,
          signal: controller.signal,
        }
      )

      if (response.status === 404) {
        throw new Error(
          'Restart the updated backend to enable explanations.'
        )
      }

      const data: unknown = await response.json()

      if (!response.ok) {
        const detail = (
          data as { detail?: unknown } | null
        )?.detail

        throw new Error(
          typeof detail === 'string'
            ? detail
            : `Explanation request failed (${response.status}).`
        )
      }

      if (
        !validExplanation(data) ||
        data.row !== point.row ||
        data.column !== point.column ||
        data.target_label !== point.prediction
      ) {
        throw new Error(
          'The backend returned an invalid explanation. Please retry.'
        )
      }

      if (
        controllerRef.current !== controller ||
        controller.signal.aborted
      ) {
        return
      }

      setResult(data)
      setShowHeatmap(true)
    } catch (cause) {
      if (controllerRef.current !== controller) return

      setError(
        timedOut
          ? 'The explanation took too long. Please retry.'
          : cause instanceof TypeError
            ? 'Cannot reach the explanation service. Check that the backend is running.'
            : cause instanceof Error
              ? cause.message
              : 'Could not explain this point.'
      )
    } finally {
      window.clearTimeout(timer)

      if (controllerRef.current === controller) {
        controllerRef.current = null
        setBusy(false)
      }
    }
  }

  return (
    <div className="xai-panel" aria-busy={busy}>
      <div className="xai-heading">
        <h3>Explain prediction</h3>
        <span className="xai-badge">XAI</span>
      </div>

      <p className="xai-copy">
        Inspect the image regions contributing to the original{' '}
        <strong>{pretty(point.prediction)}</strong> prediction.
      </p>

      <button
        type="button"
        className="xai-button"
        disabled={busy || disabled}
        onClick={() => void explain()}
      >
        {busy
          ? 'Generating explanation…'
          : result
            ? 'Regenerate explanation'
            : 'Explain prediction'}
      </button>

      {busy && (
        <p className="xai-status" role="status">
          Analyzing point {point.point}…
        </p>
      )}

      {error && (
        <p className="xai-error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <div className="xai-result">
          <div
            className="xai-views"
            role="group"
            aria-label="Explanation view"
          >
            <button
              type="button"
              aria-pressed={!showHeatmap}
              onClick={() => setShowHeatmap(false)}
            >
              Original
            </button>

            <button
              type="button"
              aria-pressed={showHeatmap}
              disabled={!result.has_signal}
              onClick={() => setShowHeatmap(true)}
            >
              Grad-CAM
            </button>
          </div>

          <figure className="xai-figure">
            <div className="xai-image">
              <img
                src={result.crop_image}
                alt={`Original crop at point ${point.point}`}
              />

              {result.has_signal &&
                showHeatmap &&
                result.heatmap_image && (
                  <img
                    className="xai-overlay"
                    src={result.heatmap_image}
                    style={{ opacity }}
                    alt={`Grad-CAM attribution for ${pretty(
                      result.target_label
                    )}`}
                  />
                )}
            </div>

            <figcaption>
              {result.has_signal
                ? 'Warmer regions show stronger positive contribution to the selected class score.'
                : 'No positive class activation was detected for this crop. Try another point.'}
            </figcaption>
          </figure>

          {result.has_signal && showHeatmap && (
            <div className="xai-opacity">
              <label htmlFor={sliderId}>
                Heatmap opacity
              </label>

              <input
                id={sliderId}
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={event =>
                  setOpacity(Number(event.target.value))
                }
              />
            </div>
          )}

          <div className="xai-meta">
            <span>
              {result.method} ·{' '}
              {pretty(result.target_label)}
            </span>

            <strong>
              {(result.target_confidence * 100).toFixed(1)}%
              {' '}confidence
            </strong>
          </div>

          {result.prediction !== result.target_label && (
            <p className="xai-copy">
              The current backend predicts{' '}
              {pretty(result.prediction)}. This explanation
              targets the saved original label.
            </p>
          )}

          <p className="xai-footnote">
            Shows model attribution, not coral boundaries.
            Human review labels remain separate.
          </p>
        </div>
      )}
    </div>
  )
}