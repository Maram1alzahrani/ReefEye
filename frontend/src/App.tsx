import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Page =
  | 'dashboard'
  | 'surveys'
  | 'analysis'
  | 'review'
  | 'sites'
  | 'directory'
  | 'exports'

type Point = {
  point: number
  row: number
  column: number
  prediction: string
  confidence: number
  needs_review: boolean
  status: string
}

type Analysis = {
  filename: string
  width: number
  height: number
  classes: string[]
  points: Point[]
  total_points: number
  review_count: number
  review_rate: number
  review_threshold: number
}

type Decision = {
  label: string
  reviewed: boolean
}

type Site = {
  id: string
  name: string
  latitude: number | null
  longitude: number | null
  createdAt: string
}

type Survey = {
  id: string
  siteId: string
  name: string
  date: string
  depth: number | null
  visibility: number | null
  notes: string
  createdAt: string
}

type Photo = {
  id: string
  surveyId: string
  filename: string
  blob: Blob
  gridSize: number
  analysis: Analysis | null
  decisions: Record<number, Decision>
  drafts: Record<number, string>
  createdAt: string
}

type Workspace = {
  version: 2
  sites: Site[]
  surveys: Survey[]
  photos: Photo[]
}

type LegacySession = {
  version: 1
  file: Blob | File | null
  result: Analysis | null
  decisions: Record<number, Decision>
  drafts?: Record<number, string>
  site: string
  gridSize: number
}

type UndoAction = {
  photoId: string
  pointId: number
  decision?: Decision
  draft?: string
}

const API = 'http://127.0.0.1:8000'

const COLORS: Record<string, string> = {
  acropora: '#0f766e',
  background: '#475569',
  galaxea: '#9a3412',
  other_coral: '#6d28d9',
  pocillopora: '#be185d',
  porites: '#1d4ed8',
  stylophora: '#4d7c0f',
}

const MODEL_CLASSES = Object.keys(COLORS)

const NAV: { id: Page; label: string }[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'surveys', label: 'Surveys' },
  { id: 'analysis', label: 'AI Analysis' },
  { id: 'review', label: 'Human Review' },
  { id: 'sites', label: 'Monitoring Sites' },
  { id: 'directory', label: 'Coral Directory' },
  { id: 'exports', label: 'Data & Exports' },
]

const PAGE_INFO: Record<
  Page,
  { title: string; description: string }
> = {
  dashboard: {
    title: 'Reef overview',
    description: 'Your sites, surveys and reef observations in one place.',
  },
  surveys: {
    title: 'Survey library',
    description: 'Keep each field visit and its photographs together.',
  },
  analysis: {
    title: 'AI Analysis',
    description: 'Explore point predictions in your reef photographs.',
  },
  review: {
    title: 'Human Review',
    description: 'Inspect uncertain predictions and confirm current labels.',
  },
  sites: {
    title: 'Monitoring sites',
    description: 'Organize reef observations by location.',
  },
  directory: {
    title: 'Coral directory',
    description: 'Explore the classification labels supported by ReefEye.',
  },
  exports: {
    title: 'Data & exports',
    description: 'Export original predictions and confirmed review labels.',
  },
}

const emptyWorkspace = (): Workspace => ({
  version: 2,
  sites: [],
  surveys: [],
  photos: [],
})

const createId = () => crypto.randomUUID()

const pretty = (value: string) =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, letter => letter.toUpperCase())

const percent = (value: number) =>
  `${(value * 100).toFixed(1)}%`

function today() {
  const date = new Date()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${date.getFullYear()}-${month}-${day}`
}

function formatDate(value: string) {
  if (!value) return 'Date not recorded'

  const date = new Date(`${value}T12:00:00`)

  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString('en-GB', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
      })
}

function currentLabel(photo: Photo, point: Point) {
  return photo.decisions[point.point]?.reviewed
    ? photo.decisions[point.point].label
    : point.prediction
}

function pointStatus(photo: Photo, point: Point) {
  if (photo.decisions[point.point]?.reviewed) {
    return 'Reviewed'
  }

  return point.needs_review
    ? 'Needs review'
    : 'Model prediction'
}

function pendingPoints(photo: Photo) {
  return (photo.analysis?.points ?? []).filter(
    point =>
      point.needs_review &&
      !photo.decisions[point.point]?.reviewed
  )
}

function reviewedCount(photo: Photo) {
  return (photo.analysis?.points ?? []).filter(
    point => photo.decisions[point.point]?.reviewed
  ).length
}

function labelCounts(photos: Photo[]) {
  return photos.reduce<Record<string, number>>(
    (counts, photo) => {
      for (const point of photo.analysis?.points ?? []) {
        const label = currentLabel(photo, point)
        counts[label] = (counts[label] ?? 0) + 1
      }

      return counts
    },
    {}
  )
}

function validAnalysis(value: unknown): value is Analysis {
  if (!value || typeof value !== 'object') return false

  const data = value as Analysis

  return (
    typeof data.filename === 'string' &&
    Number.isFinite(data.width) &&
    data.width > 0 &&
    Number.isFinite(data.height) &&
    data.height > 0 &&
    Array.isArray(data.classes) &&
    data.classes.length > 0 &&
    data.classes.every(label => typeof label === 'string') &&
    Number.isFinite(data.review_threshold) &&
    data.review_threshold >= 0 &&
    data.review_threshold <= 1 &&
    Array.isArray(data.points) &&
    data.points.length > 0 &&
    data.points.every(
      point =>
        point !== null &&
        typeof point === 'object' &&
        Number.isInteger(point.point) &&
        Number.isFinite(point.row) &&
        point.row >= 0 &&
        point.row < data.height &&
        Number.isFinite(point.column) &&
        point.column >= 0 &&
        point.column < data.width &&
        data.classes.includes(point.prediction) &&
        Number.isFinite(point.confidence) &&
        point.confidence >= 0 &&
        point.confidence <= 1 &&
        typeof point.needs_review === 'boolean'
    ) &&
    new Set(data.points.map(point => point.point)).size ===
      data.points.length
  )
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('reefeye-session', 2)
    let blocked = false

    request.onupgradeneeded = () => {
      const database = request.result

      if (!database.objectStoreNames.contains('workspace')) {
        database.createObjectStore('workspace')
      }
    }

    request.onerror = () => reject(request.error)

    request.onblocked = () => {
      blocked = true
      reject(
        new Error('Close other ReefEye tabs and reload this page.')
      )
    }

    request.onsuccess = () => {
      if (blocked) {
        request.result.close()
        return
      }

      const database = request.result
      database.onversionchange = () => database.close()
      resolve(database)
    }
  })
}

function migrateLegacy(
  saved: LegacySession | undefined
): Workspace {
  const workspace = emptyWorkspace()

  if (
    !saved ||
    saved.version !== 1 ||
    !(saved.file instanceof Blob) ||
    (saved.result && !validAnalysis(saved.result))
  ) {
    return workspace
  }

  const siteId = createId()
  const surveyId = createId()
  const createdAt = new Date().toISOString()

  workspace.sites.push({
    id: siteId,
    name: saved.site?.trim() || 'Imported site',
    latitude: null,
    longitude: null,
    createdAt,
  })

  workspace.surveys.push({
    id: surveyId,
    siteId,
    name: 'Imported session',
    date: '',
    depth: null,
    visibility: null,
    notes: 'Imported from the previous ReefEye workspace.',
    createdAt,
  })

  workspace.photos.push({
    id: createId(),
    surveyId,
    filename:
      saved.result?.filename ||
      (saved.file instanceof File ? saved.file.name : 'reef-image'),
    blob: saved.file,
    gridSize: saved.gridSize || 5,
    analysis: saved.result,
    decisions: saved.decisions ?? {},
    drafts: saved.drafts ?? {},
    createdAt,
  })

  return workspace
}

async function readWorkspace(): Promise<Workspace> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const stores = ['workspace']

    if (database.objectStoreNames.contains('session')) {
      stores.push('session')
    }

    const transaction = database.transaction(stores, 'readonly')

    const workspaceRequest = transaction
      .objectStore('workspace')
      .get('current')

    const legacyRequest = stores.includes('session')
      ? transaction.objectStore('session').get('current')
      : null

    transaction.oncomplete = () => {
      const saved = workspaceRequest.result as Workspace | undefined
      const legacy = legacyRequest?.result as LegacySession | undefined

      database.close()

      if (saved) {
        if (
          saved.version !== 2 ||
          !Array.isArray(saved.sites) ||
          !Array.isArray(saved.surveys) ||
          !Array.isArray(saved.photos)
        ) {
          reject(
            new Error('The saved workspace could not be opened.')
          )
          return
        }

        resolve(saved)
        return
      }

      resolve(migrateLegacy(legacy))
    }

    transaction.onerror = () => {
      database.close()
      reject(transaction.error)
    }

    transaction.onabort = () => {
      database.close()
      reject(transaction.error)
    }
  })
}

async function saveWorkspace(
  workspace: Workspace
): Promise<void> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(
      'workspace',
      'readwrite'
    )

    transaction.objectStore('workspace').put(workspace, 'current')

    transaction.oncomplete = () => {
      database.close()
      resolve()
    }

    transaction.onerror = () => {
      database.close()
      reject(transaction.error)
    }

    transaction.onabort = () => {
      database.close()
      reject(transaction.error)
    }
  })
}

function usePhotoUrl(blob: Blob | null | undefined) {
  const [value, setValue] = useState<{
    blob: Blob
    url: string
  } | null>(null)

  useEffect(() => {
    if (!blob) {
      setValue(null)
      return
    }

    const url = URL.createObjectURL(blob)
    setValue({ blob, url })

    return () => URL.revokeObjectURL(url)
  }, [blob])

  return value && value.blob === blob ? value.url : ''
}

function PhotoImage({
  photo,
  className = '',
}: {
  photo: Photo
  className?: string
}) {
  const url = usePhotoUrl(photo.blob)

  return url ? (
    <img
      className={className}
      src={url}
      alt={photo.filename}
    />
  ) : (
    <div className={`${className} photo-loading`}>
      Loading image…
    </div>
  )
}

function PointCrop({
  photo,
  point,
}: {
  photo: Photo
  point: Point
}) {
  const url = usePhotoUrl(photo.blob)

  if (!url || !photo.analysis) {
    return (
      <div className="crop-placeholder">
        Loading crop…
      </div>
    )
  }

  return (
    <svg
      className="point-crop"
      viewBox={`${point.column - 112} ${point.row - 112} 224 224`}
      role="img"
      aria-label={`Image crop around point ${point.point}`}
    >
      <image
        href={url}
        width={photo.analysis.width}
        height={photo.analysis.height}
      />

      <circle
        cx={point.column}
        cy={point.row}
        r={5}
        fill="none"
        stroke="white"
        strokeWidth={2}
      />

      <path
        d={
          `M ${point.column - 11} ${point.row} H ${point.column + 11} ` +
          `M ${point.column} ${point.row - 11} V ${point.row + 11}`
        }
        stroke="#fbbf24"
        strokeWidth={2}
      />
    </svg>
  )
}

function Distribution({ photos }: { photos: Photo[] }) {
  const counts = labelCounts(photos)

  const entries = Object.entries(counts).sort(
    (a, b) => b[1] - a[1]
  )

  const total = entries.reduce(
    (sum, [, count]) => sum + count,
    0
  )

  let end = 0

  const slices = entries.map(([label, count]) => {
    const start = end
    end += (count / total) * 100

    return `${COLORS[label] ?? '#475569'} ${start}% ${end}%`
  })

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Classification distribution</h2>
          <p>
            Current labels, including confirmed corrections.
          </p>
        </div>
      </div>

      {total ? (
        <>
          <div className="distribution-layout">
            <div
              className="donut"
              role="img"
              aria-label={entries
                .map(
                  ([label, count]) =>
                    `${pretty(label)}: ${count}`
                )
                .join(', ')}
              style={{
                background: `conic-gradient(${slices.join(',')})`,
              }}
            >
              <div>
                <strong>{total}</strong>
                <span>sample points</span>
              </div>
            </div>

            <div className="distribution-list">
              {entries.map(([label, count]) => (
                <div
                  className="distribution-row"
                  key={label}
                >
                  <span className="color-label">
                    <i
                      style={{
                        background: COLORS[label] ?? '#475569',
                      }}
                    />
                    {pretty(label)}
                  </span>

                  <strong>{percent(count / total)}</strong>
                  <span>{count}</span>
                </div>
              ))}
            </div>
          </div>

          <p className="footnote">
            Percentages describe sampled points, not coral
            area coverage.
          </p>
        </>
      ) : (
        <div className="empty compact">
          Analyze a photograph to see the label distribution.
        </div>
      )}
    </section>
  )
}

function SiteMap({
  sites,
  selectedId,
  onSelect,
}: {
  sites: Site[]
  selectedId: string
  onSelect: (siteId: string) => void
}) {
  const mappedSites = sites.filter(
    site =>
      site.latitude !== null &&
      site.longitude !== null
  )

  const selected =
    mappedSites.find(site => site.id === selectedId) ??
    mappedSites[0]

  let mapUrl = ''

  if (selected) {
    const latitude = selected.latitude!
    const longitude = selected.longitude!

    const bbox = [
      Math.max(-180, longitude - 0.18),
      Math.max(-85, latitude - 0.12),
      Math.min(180, longitude + 0.18),
      Math.min(85, latitude + 0.12),
    ].join(',')

    mapUrl =
      'https://www.openstreetmap.org/export/embed.html?' +
      `bbox=${encodeURIComponent(bbox)}&layer=mapnik&marker=` +
      encodeURIComponent(`${latitude},${longitude}`)
  }

  return (
    <section className="panel map-panel">
      <div className="panel-heading">
        <div>
          <h2>Monitoring sites</h2>
          <p>Locations recorded in your workspace.</p>
        </div>

        {mappedSites.length > 0 && (
          <select
            className="input map-select"
            aria-label="Site shown on the map"
            value={selected?.id ?? ''}
            onChange={event => onSelect(event.target.value)}
          >
            {mappedSites.map(site => (
              <option key={site.id} value={site.id}>
                {site.name}
              </option>
            ))}
          </select>
        )}
      </div>

      {selected ? (
        <>
          <iframe
            className="site-map"
            title={`Map of ${selected.name}`}
            src={mapUrl}
            loading="lazy"
          />

          <div className="map-caption">
            <strong>{selected.name}</strong>
            <span>
              {selected.latitude!.toFixed(5)},{' '}
              {selected.longitude!.toFixed(5)}
            </span>
          </div>
        </>
      ) : (
        <div className="map-empty">
          <span className="eyebrow">LOCATION</span>
          <h3>Your reef sites belong here.</h3>
          <p>
            Add a site with latitude and longitude to display
            its location on the map.
          </p>
        </div>
      )}

      {sites.length > mappedSites.length && (
        <p className="footnote">
          {sites.length - mappedSites.length} site(s) have
          no recorded coordinates.
        </p>
      )}
    </section>
  )
}

function App() {
  const [workspace, setWorkspace] =
    useState<Workspace>(emptyWorkspace)

  const [ready, setReady] = useState(false)
  const [page, setPage] = useState<Page>('dashboard')
  const [connection, setConnection] = useState('Checking model…')
  const [saveStatus, setSaveStatus] = useState('Opening workspace…')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const [activeSurveyId, setActiveSurveyId] = useState('')
  const [activePhotoId, setActivePhotoId] = useState('')
  const [selectedPointId, setSelectedPointId] =
    useState<number | null>(null)

  const [mapSiteId, setMapSiteId] = useState('')
  const [siteFilter, setSiteFilter] = useState('all')
  const [query, setQuery] = useState('')

  const [showSiteForm, setShowSiteForm] = useState(false)
  const [showSurveyForm, setShowSurveyForm] = useState(false)
  const [newSurveyFiles, setNewSurveyFiles] = useState<File[]>([])

  const [showMarkers, setShowMarkers] = useState(true)

  const [markerMode, setMarkerMode] = useState<
    'number' | 'confidence' | 'label'
  >('number')

  const [classFilter, setClassFilter] = useState('all')
  const [pendingOnly, setPendingOnly] = useState(false)
  const [busyPhotoId, setBusyPhotoId] =
    useState<string | null>(null)

  const [undo, setUndo] = useState<UndoAction | null>(null)

  const controllerRef = useRef<AbortController | null>(null)
  const activePhotoRef = useRef('')
  const saveQueue = useRef<Promise<void>>(Promise.resolve())
  const revisionRef = useRef(0)
  const unsavedRef = useRef(false)
  const storageEnabledRef = useRef(true)

  useEffect(() => {
    let active = true

    readWorkspace()
      .then(saved => {
        if (!active) return

        setWorkspace(saved)

        if (saved.surveys.length) {
          setActiveSurveyId(saved.surveys[0].id)
        }

        if (saved.photos.length) {
          setMessage('Your saved workspace is ready.')
        }
      })
      .catch(caught => {
        if (!active) return

        storageEnabledRef.current = false
        setSaveStatus('Local storage unavailable')

        setError(
          caught instanceof Error
            ? caught.message
            : 'Could not open local storage. Export any new results before leaving.'
        )
      })
      .finally(() => {
        if (active) setReady(true)
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!ready || !storageEnabledRef.current) return

    unsavedRef.current = true
    setSaveStatus('Saving on this device…')

    const revision = ++revisionRef.current

    const timer = window.setTimeout(() => {
      saveQueue.current = saveQueue.current
        .catch(() => {})
        .then(async () => {
          try {
            await saveWorkspace(workspace)

            if (revision === revisionRef.current) {
              unsavedRef.current = false
              setSaveStatus('Saved on this device')
            }
          } catch {
            if (revision === revisionRef.current) {
              setSaveStatus(
                'Not saved — export results before leaving'
              )
            }
          }
        })
    }, 300)

    return () => window.clearTimeout(timer)
  }, [workspace, ready])

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (!unsavedRef.current) return
      event.preventDefault()
      event.returnValue = ''
    }

    window.addEventListener('beforeunload', warn)

    return () =>
      window.removeEventListener('beforeunload', warn)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    let active = true

    const timer = window.setTimeout(
      () => controller.abort(),
      8000
    )

    fetch(`${API}/health`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error('Unavailable')

        const health = await response.json()

        if (active) {
          setConnection(
            health.status === 'ready'
              ? 'Model connected'
              : 'Model unavailable'
          )
        }
      })
      .catch(() => {
        if (active) setConnection('Model not connected')
      })
      .finally(() => window.clearTimeout(timer))

    return () => {
      active = false
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    return () => controllerRef.current?.abort()
  }, [])

  const activeSurvey =
    workspace.surveys.find(
      survey => survey.id === activeSurveyId
    ) ?? workspace.surveys[0]

  const activeSite = workspace.sites.find(
    site => site.id === activeSurvey?.siteId
  )

  const surveyPhotos = workspace.photos.filter(
    photo => photo.surveyId === activeSurvey?.id
  )

  const activePhoto =
    surveyPhotos.find(photo => photo.id === activePhotoId) ??
    surveyPhotos[0]

  activePhotoRef.current = activePhoto?.id ?? ''

  const preview = usePhotoUrl(activePhoto?.blob)
  const points = activePhoto?.analysis?.points ?? []
  const pending = activePhoto ? pendingPoints(activePhoto) : []

  const activePoint =
    points.find(point => point.point === selectedPointId) ??
    pending[0] ??
    points[0]

  const analyzedPhotos = workspace.photos.filter(
    photo => photo.analysis
  )

  const totalPending = workspace.photos.reduce(
    (sum, photo) => sum + pendingPoints(photo).length,
    0
  )

  const totalReviewed = workspace.photos.reduce(
    (sum, photo) => sum + reviewedCount(photo),
    0
  )

  const totalPoints = analyzedPhotos.reduce(
    (sum, photo) =>
      sum + (photo.analysis?.points.length ?? 0),
    0
  )

  const visibleSurveys = workspace.surveys
    .filter(survey => {
      const site = workspace.sites.find(
        item => item.id === survey.siteId
      )

      return (
        (siteFilter === 'all' || survey.siteId === siteFilter) &&
        `${survey.name} ${site?.name ?? ''} ${survey.date}`
          .toLowerCase()
          .includes(query.trim().toLowerCase())
      )
    })
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))

  function navigate(next: Page) {
    setPage(next)
    setError('')
    window.scrollTo({ top: 0, behavior: 'auto' })
  }

  function openSurvey(
    surveyId: string,
    destination: Page = 'analysis'
  ) {
    const firstPhoto = workspace.photos.find(
      photo => photo.surveyId === surveyId
    )

    setActiveSurveyId(surveyId)
    setActivePhotoId(firstPhoto?.id ?? '')
    setSelectedPointId(null)
    setClassFilter('all')
    setPendingOnly(destination === 'review')
    setShowMarkers(true)
    navigate(destination)
  }

  function selectPhoto(photoId: string) {
    setActivePhotoId(photoId)
    setSelectedPointId(null)
    setClassFilter('all')
  }

  function updatePhoto(
    photoId: string,
    update: (photo: Photo) => Photo
  ) {
    setWorkspace(previous => ({
      ...previous,
      photos: previous.photos.map(photo =>
        photo.id === photoId ? update(photo) : photo
      ),
    }))
  }

  function validateFiles(files: File[]) {
    if (
      files.some(
        file =>
          !['image/jpeg', 'image/png', 'image/webp'].includes(
            file.type
          )
      )
    ) {
      setError('Please choose JPG, PNG or WebP photographs.')
      return false
    }

    if (files.some(file => file.size > 10 * 1024 * 1024)) {
      setError('Each photograph must be 10 MB or smaller.')
      return false
    }

    return true
  }

  function makePhotos(
    files: File[],
    surveyId: string
  ): Photo[] {
    return files.map(file => ({
      id: createId(),
      surveyId,
      filename: file.name,
      blob: file,
      gridSize: 5,
      analysis: null,
      decisions: {},
      drafts: {},
      createdAt: new Date().toISOString(),
    }))
  }

  function addPhotos(files: File[]) {
    if (
      !activeSurvey ||
      !files.length ||
      !validateFiles(files)
    ) {
      return
    }

    const added = makePhotos(files, activeSurvey.id)

    setWorkspace(previous => ({
      ...previous,
      photos: [...previous.photos, ...added],
    }))

    setActivePhotoId(added[0].id)
    setSelectedPointId(null)
    setError('')

    setMessage(
      `${added.length} photograph(s) added to ${activeSurvey.name}.`
    )
  }

  function createSite(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const data = new FormData(event.currentTarget)
    const name = String(data.get('name') ?? '').trim()
    const latitudeText = String(
      data.get('latitude') ?? ''
    ).trim()
    const longitudeText = String(
      data.get('longitude') ?? ''
    ).trim()

    if (!name) return

    if (Boolean(latitudeText) !== Boolean(longitudeText)) {
      setError(
        'Enter both latitude and longitude, or leave both empty.'
      )
      return
    }

    const latitude = latitudeText ? Number(latitudeText) : null
    const longitude = longitudeText ? Number(longitudeText) : null

    if (
      (latitude !== null &&
        (!Number.isFinite(latitude) ||
          latitude < -85 ||
          latitude > 85)) ||
      (longitude !== null &&
        (!Number.isFinite(longitude) ||
          longitude < -180 ||
          longitude > 180))
    ) {
      setError(
        'Enter valid coordinates: latitude −85 to 85 and longitude −180 to 180.'
      )
      return
    }

    if (
      workspace.sites.some(
        site => site.name.toLowerCase() === name.toLowerCase()
      )
    ) {
      setError('A site with this name already exists.')
      return
    }

    const site: Site = {
      id: createId(),
      name,
      latitude,
      longitude,
      createdAt: new Date().toISOString(),
    }

    setWorkspace(previous => ({
      ...previous,
      sites: [...previous.sites, site],
    }))

    setMapSiteId(site.id)
    setShowSiteForm(false)
    setError('')
    setMessage(`${name} added to monitoring sites.`)
  }

  function createSurvey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const data = new FormData(event.currentTarget)
    const siteId = String(data.get('siteId') ?? '')
    const name = String(data.get('name') ?? '').trim()
    const date = String(data.get('date') ?? '')

    if (
      !name ||
      !date ||
      !workspace.sites.some(site => site.id === siteId)
    ) {
      setError(
        'Choose a site and enter the survey name and date.'
      )
      return
    }

    if (!validateFiles(newSurveyFiles)) return

    const depthText = String(data.get('depth') ?? '').trim()
    const visibilityText = String(
      data.get('visibility') ?? ''
    ).trim()

    const depth = depthText ? Number(depthText) : null
    const visibility = visibilityText
      ? Number(visibilityText)
      : null

    if (
      (depth !== null &&
        (!Number.isFinite(depth) || depth < 0)) ||
      (visibility !== null &&
        (!Number.isFinite(visibility) || visibility < 0))
    ) {
      setError(
        'Depth and visibility must be non-negative numbers.'
      )
      return
    }

    const survey: Survey = {
      id: createId(),
      siteId,
      name,
      date,
      depth,
      visibility,
      notes: String(data.get('notes') ?? '').trim(),
      createdAt: new Date().toISOString(),
    }

    const photos = makePhotos(newSurveyFiles, survey.id)

    setWorkspace(previous => ({
      ...previous,
      surveys: [...previous.surveys, survey],
      photos: [...previous.photos, ...photos],
    }))

    setActiveSurveyId(survey.id)
    setActivePhotoId(photos[0]?.id ?? '')
    setSelectedPointId(null)
    setShowSurveyForm(false)
    setNewSurveyFiles([])
    setError('')
    setMessage(`${survey.name} created.`)
    navigate('analysis')
  }

  async function analyzePhoto(photo: Photo) {
    if (controllerRef.current) return

    if (
      photo.analysis &&
      !window.confirm(
        'Reanalyze this photograph? Its current predictions and review decisions will be replaced after a successful analysis.'
      )
    ) {
      return
    }

    const controller = new AbortController()
    controllerRef.current = controller

    setBusyPhotoId(photo.id)
    setError('')
    setMessage(`Analyzing ${photo.filename}…`)

    const body = new FormData()
    body.append('file', photo.blob, photo.filename)
    body.append('grid_size', String(photo.gridSize))

    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        body,
        signal: controller.signal,
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => null)

        throw new Error(
          typeof detail?.detail === 'string'
            ? detail.detail
            : `Analysis failed (${response.status}).`
        )
      }

      const result: unknown = await response.json()

      if (!validAnalysis(result)) {
        throw new Error('The analysis response is incomplete.')
      }

      if (controller.signal.aborted) return

      updatePhoto(photo.id, current => ({
        ...current,
        analysis: result,
        decisions: {},
        drafts: {},
      }))

      if (activePhotoRef.current === photo.id) {
        setSelectedPointId(
          result.points.find(point => point.needs_review)?.point ??
            result.points[0]?.point ??
            null
        )
      }

      setUndo(null)
      setConnection('Model connected')

      setMessage(
        `${photo.filename}: ${result.points.length} points analyzed.`
      )
    } catch (caught) {
      if (controller.signal.aborted) {
        setMessage(
          'Analysis cancelled. Existing results were kept.'
        )
      } else {
        if (caught instanceof TypeError) {
          setConnection('Model not connected')
        }

        setError(
          caught instanceof TypeError
            ? 'The analysis service is unavailable. Check that it is running and try again.'
            : caught instanceof Error
              ? caught.message
              : 'Analysis failed. Please retry.'
        )

        setMessage('')
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setBusyPhotoId(null)
      }
    }
  }

  function chooseLabel(
    photo: Photo,
    point: Point,
    label: string
  ) {
    updatePhoto(photo.id, current => ({
      ...current,
      drafts: {
        ...current.drafts,
        [point.point]: label,
      },
    }))
  }

  function discardDraft(photo: Photo, point: Point) {
    updatePhoto(photo.id, current => {
      const drafts = { ...current.drafts }
      delete drafts[point.point]
      return { ...current, drafts }
    })
  }

  function confirmPoint(
    photo: Photo,
    point: Point,
    advance: boolean
  ) {
    if (busyPhotoId === photo.id) return

    const label =
      photo.drafts[point.point] ??
      photo.decisions[point.point]?.label ??
      point.prediction

    setUndo({
      photoId: photo.id,
      pointId: point.point,
      decision: photo.decisions[point.point],
      draft: photo.drafts[point.point],
    })

    updatePhoto(photo.id, current => {
      const drafts = { ...current.drafts }
      delete drafts[point.point]

      return {
        ...current,
        drafts,
        decisions: {
          ...current.decisions,
          [point.point]: {
            label,
            reviewed: true,
          },
        },
      }
    })

    setMessage(
      `Point ${point.point} confirmed as ${pretty(label)}.`
    )

    if (advance) {
      const remaining = pendingPoints(photo).filter(
        item => item.point !== point.point
      )

      const next =
        remaining.find(item => item.point > point.point) ??
        remaining[0]

      if (next) {
        setSelectedPointId(next.point)
      } else {
        setMessage(
          'All flagged points in this image have been reviewed.'
        )
      }
    }
  }

  function undoLastReview() {
    if (!undo) return

    const photo = workspace.photos.find(
      item => item.id === undo.photoId
    )

    if (!photo) return

    updatePhoto(photo.id, current => {
      const decisions = { ...current.decisions }
      const drafts = { ...current.drafts }

      if (undo.decision) {
        decisions[undo.pointId] = undo.decision
      } else {
        delete decisions[undo.pointId]
      }

      if (undo.draft !== undefined) {
        drafts[undo.pointId] = undo.draft
      } else {
        delete drafts[undo.pointId]
      }

      return { ...current, decisions, drafts }
    })

    openSurvey(photo.surveyId, 'review')
    setActivePhotoId(photo.id)
    setSelectedPointId(undo.pointId)
    setPendingOnly(false)
    setMessage(`Review for point ${undo.pointId} undone.`)
    setUndo(null)
  }

  function openNextPending() {
    const photo = workspace.photos.find(
      item => pendingPoints(item).length > 0
    )

    if (!photo) {
      setMessage(
        'There are no flagged points awaiting review.'
      )
      navigate('review')
      return
    }

    openSurvey(photo.surveyId, 'review')
    setActivePhotoId(photo.id)
    setSelectedPointId(pendingPoints(photo)[0].point)
  }

  function exportPhotos(
    photos: Photo[],
    filename: string
  ) {
    const selected = photos.filter(photo => photo.analysis)

    if (!selected.length) {
      setError(
        'Analyze at least one photograph before exporting.'
      )
      return
    }

    if (
      selected.some(
        photo => Object.keys(photo.drafts).length > 0
      ) &&
      !window.confirm(
        'Unconfirmed selections are excluded. Export confirmed labels and unchanged model predictions?'
      )
    ) {
      return
    }

    const cell = (value: unknown) => {
      let text =
        value === null || value === undefined
          ? ''
          : String(value)

      if (/^\s*[=+@-]/.test(text)) {
        text = `'${text}`
      }

      return `"${text.replace(/"/g, '""')}"`
    }

    const rows: unknown[][] = [[
      'site',
      'latitude',
      'longitude',
      'survey',
      'survey_date',
      'depth_m',
      'visibility_m',
      'image',
      'point',
      'row',
      'column',
      'model_prediction',
      'model_confidence',
      'model_needs_review',
      'review_threshold',
      'current_label',
      'review_status',
    ]]

    for (const photo of selected) {
      const survey = workspace.surveys.find(
        item => item.id === photo.surveyId
      )

      const site = workspace.sites.find(
        item => item.id === survey?.siteId
      )

      for (const point of photo.analysis!.points) {
        rows.push([
          site?.name,
          site?.latitude,
          site?.longitude,
          survey?.name,
          survey?.date,
          survey?.depth,
          survey?.visibility,
          photo.filename,
          point.point,
          point.row,
          point.column,
          point.prediction,
          point.confidence,
          point.needs_review,
          photo.analysis!.review_threshold,
          currentLabel(photo, point),
          pointStatus(photo, point),
        ])
      }
    }

    const blob = new Blob(
      [
        '\uFEFF' +
          rows
            .map(row => row.map(cell).join(','))
            .join('\r\n'),
      ],
      { type: 'text/csv;charset=utf-8;' }
    )

    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')

    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()

    window.setTimeout(
      () => URL.revokeObjectURL(url),
      1000
    )

    setMessage(
      `CSV export requested for ${selected.length} analyzed image(s).`
    )
    setError('')
  }

  function renderSurveyCard(survey: Survey) {
    const site = workspace.sites.find(
      item => item.id === survey.siteId
    )

    const photos = workspace.photos.filter(
      photo => photo.surveyId === survey.id
    )

    const analyzed = photos.filter(
      photo => photo.analysis
    ).length

    const needsReview = photos.reduce(
      (sum, photo) => sum + pendingPoints(photo).length,
      0
    )

    return (
      <article className="survey-card" key={survey.id}>
        <button
          className="survey-cover-button"
          onClick={() => openSurvey(survey.id)}
          aria-label={`Open ${survey.name}`}
        >
          {photos[0] ? (
            <PhotoImage
              photo={photos[0]}
              className="survey-cover"
            />
          ) : (
            <div className="survey-cover empty-cover">
              No photographs yet
            </div>
          )}
        </button>

        <div className="survey-card-body">
          <div className="card-title-row">
            <span className="eyebrow">
              {site?.name ?? 'Unknown site'}
            </span>

            <span
              className={`badge ${needsReview ? 'warning' : ''}`}
            >
              {needsReview
                ? `${needsReview} pending`
                : `${analyzed} analyzed`}
            </span>
          </div>

          <h3>{survey.name}</h3>
          <p>{formatDate(survey.date)}</p>

          <div className="survey-card-footer">
            <span>{photos.length} photographs</span>
            <button
              className="text-button"
              onClick={() => openSurvey(survey.id)}
            >
              Open survey →
            </button>
          </div>
        </div>
      </article>
    )
  }

  function renderDashboard() {
    const recent = [...workspace.surveys]
      .sort((a, b) =>
        b.createdAt.localeCompare(a.createdAt)
      )
      .slice(0, 4)

    return (
      <>
        <section className="ocean-banner">
          <div>
            <span className="eyebrow">
              REEFEYE OBSERVATORY
            </span>
            <h2>A closer look beneath the surface.</h2>
            <p>
              Follow your reef observations from photograph
              to review.
            </p>

            <button
              className="button light"
              onClick={() =>
                workspace.sites.length
                  ? setShowSurveyForm(true)
                  : setShowSiteForm(true)
              }
            >
              {workspace.sites.length
                ? 'Start a survey'
                : 'Add your first site'}
            </button>
          </div>
        </section>

        <div className="metrics-grid">
          <div className="metric">
            <span>Monitoring sites</span>
            <strong>{workspace.sites.length}</strong>
            <small>Recorded locations</small>
          </div>

          <div className="metric">
            <span>Analyzed images</span>
            <strong>{analyzedPhotos.length}</strong>
            <small>
              {workspace.photos.length} photographs in your library
            </small>
          </div>

          <div className="metric">
            <span>Reviewed points</span>
            <strong>{totalReviewed}</strong>
            <small>Confirmed by a reviewer</small>
          </div>

          <button
            className="metric pending-metric"
            onClick={openNextPending}
          >
            <span>Awaiting review</span>
            <strong>{totalPending}</strong>
            <small>Open the next flagged point →</small>
          </button>
        </div>

        <div className="dashboard-grid">
          <SiteMap
            sites={workspace.sites}
            selectedId={mapSiteId}
            onSelect={setMapSiteId}
          />

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>Recent surveys</h2>
                <p>Your latest field observations.</p>
              </div>

              <button
                className="text-button"
                onClick={() => navigate('surveys')}
              >
                View all →
              </button>
            </div>

            {recent.length ? (
              <div className="recent-surveys">
                {recent.map(survey => {
                  const site = workspace.sites.find(
                    item => item.id === survey.siteId
                  )

                  const photos = workspace.photos.filter(
                    photo => photo.surveyId === survey.id
                  )

                  return (
                    <button
                      className="recent-survey"
                      key={survey.id}
                      onClick={() => openSurvey(survey.id)}
                    >
                      {photos[0] ? (
                        <PhotoImage
                          photo={photos[0]}
                          className="recent-thumb"
                        />
                      ) : (
                        <div className="recent-thumb empty-cover">
                          —
                        </div>
                      )}

                      <span>
                        <strong>{survey.name}</strong>
                        <small>
                          {site?.name} · {formatDate(survey.date)}
                        </small>
                      </span>

                      <span className="recent-count">
                        {photos.length} images
                      </span>
                    </button>
                  )
                })}
              </div>
            ) : (
              <div className="empty compact">
                Create a site, then start your first survey.
              </div>
            )}
          </section>
        </div>

        <Distribution photos={workspace.photos} />
      </>
    )
  }

  function renderSurveys() {
    return (
      <>
        <div className="toolbar panel">
          <input
            className="input search-input"
            placeholder="Search surveys, sites or dates…"
            aria-label="Search surveys"
            value={query}
            onChange={event => setQuery(event.target.value)}
          />

          <select
            className="input"
            aria-label="Filter surveys by site"
            value={siteFilter}
            onChange={event =>
              setSiteFilter(event.target.value)
            }
          >
            <option value="all">All sites</option>

            {workspace.sites.map(site => (
              <option key={site.id} value={site.id}>
                {site.name}
              </option>
            ))}
          </select>

          <button
            className="button primary"
            onClick={() =>
              workspace.sites.length
                ? setShowSurveyForm(true)
                : setShowSiteForm(true)
            }
          >
            {workspace.sites.length
              ? 'New survey'
              : 'Add a site first'}
          </button>
        </div>

        {visibleSurveys.length ? (
          <div className="survey-grid">
            {visibleSurveys.map(renderSurveyCard)}
          </div>
        ) : (
          <section className="panel empty">
            <h2>No surveys to display</h2>
            <p>Create a survey or change the search filters.</p>
          </section>
        )}
      </>
    )
  }

  function renderSites() {
    return (
      <>
        <div className="section-title">
          <p>{workspace.sites.length} recorded sites</p>

          <button
            className="button primary"
            onClick={() => setShowSiteForm(true)}
          >
            Add monitoring site
          </button>
        </div>

        <SiteMap
          sites={workspace.sites}
          selectedId={mapSiteId}
          onSelect={setMapSiteId}
        />

        <div className="site-grid">
          {workspace.sites.map(site => {
            const surveys = workspace.surveys.filter(
              survey => survey.siteId === site.id
            )

            const surveyIds = new Set(
              surveys.map(survey => survey.id)
            )

            const photos = workspace.photos.filter(
              photo => surveyIds.has(photo.surveyId)
            )

            return (
              <article
                className="panel site-card"
                key={site.id}
              >
                <span className="eyebrow">
                  MONITORING SITE
                </span>

                <h2>{site.name}</h2>

                <p>
                  {site.latitude !== null &&
                  site.longitude !== null
                    ? `${site.latitude.toFixed(5)}, ${site.longitude.toFixed(5)}`
                    : 'Coordinates not recorded'}
                </p>

                <div className="site-numbers">
                  <div>
                    <strong>{surveys.length}</strong>
                    <span>Surveys</span>
                  </div>

                  <div>
                    <strong>{photos.length}</strong>
                    <span>Images</span>
                  </div>

                  <div>
                    <strong>
                      {
                        photos.filter(photo => photo.analysis)
                          .length
                      }
                    </strong>
                    <span>Analyzed</span>
                  </div>
                </div>

                <button
                  className="button secondary full"
                  onClick={() => {
                    setSiteFilter(site.id)
                    setQuery('')
                    navigate('surveys')
                  }}
                >
                  View site surveys
                </button>
              </article>
            )
          })}
        </div>
      </>
    )
  }

  function renderImageCanvas() {
    if (!activePhoto || !preview) {
      return (
        <div className="image-empty">
          Choose a photograph to begin.
        </div>
      )
    }

    const analysis = activePhoto.analysis

    if (!analysis) {
      return (
        <img
          className="unprocessed-photo"
          src={preview}
          alt={activePhoto.filename}
        />
      )
    }

    const radius =
      Math.max(analysis.width, analysis.height) * 0.013

    const visible = analysis.points.filter(
      point =>
        (classFilter === 'all' ||
          currentLabel(activePhoto, point) === classFilter) &&
        (!pendingOnly ||
          (point.needs_review &&
            !activePhoto.decisions[point.point]?.reviewed))
    )

    return (
      <svg
        className="analysis-canvas"
        viewBox={`0 0 ${analysis.width} ${analysis.height}`}
        aria-label="Reef photograph with interactive classification points"
      >
        <image
          href={preview}
          width={analysis.width}
          height={analysis.height}
        />

        {showMarkers &&
          visible.map(point => {
            const label = currentLabel(activePhoto, point)
            const reviewed =
              activePhoto.decisions[point.point]?.reviewed
            const selected =
              activePoint?.point === point.point
            const needsReview =
              point.needs_review && !reviewed

            return (
              <g
                key={point.point}
                role="button"
                className="point-marker"
                tabIndex={0}
                aria-pressed={selected}
                aria-label={`Point ${point.point}: ${pretty(label)}, ${percent(point.confidence)}, ${pointStatus(activePhoto, point)}`}
                onClick={() =>
                  setSelectedPointId(point.point)
                }
                onKeyDown={event => {
                  if (
                    event.key === 'Enter' ||
                    event.key === ' '
                  ) {
                    event.preventDefault()
                    setSelectedPointId(point.point)
                  }
                }}
              >
                <title>
                  {`Point ${point.point}: ${pretty(label)} · ${pointStatus(activePhoto, point)}`}
                </title>

                <circle
                  cx={point.column}
                  cy={point.row}
                  r={radius * 1.7}
                  fill="transparent"
                />

                {selected && (
                  <circle
                    cx={point.column}
                    cy={point.row}
                    r={radius * 1.45}
                    fill="none"
                    stroke="white"
                    strokeWidth={Math.max(2, radius * 0.16)}
                  />
                )}

                <circle
                  cx={point.column}
                  cy={point.row}
                  r={
                    markerMode === 'confidence'
                      ? radius * 1.3
                      : radius
                  }
                  fill={COLORS[label] ?? '#475569'}
                  stroke={
                    needsReview
                      ? '#fbbf24'
                      : reviewed
                        ? '#93c5fd'
                        : 'white'
                  }
                  strokeWidth={Math.max(2, radius * 0.16)}
                />

                <text
                  x={point.column}
                  y={point.row}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill="white"
                  fontSize={radius * 0.9}
                  fontWeight="700"
                  pointerEvents="none"
                >
                  {markerMode === 'confidence'
                    ? `${Math.round(point.confidence * 100)}%`
                    : point.point}
                </text>

                {(needsReview || reviewed) && (
                  <text
                    x={point.column + radius}
                    y={point.row - radius}
                    fill="white"
                    stroke="#102e43"
                    strokeWidth={radius * 0.18}
                    paintOrder="stroke"
                    fontSize={radius * 1.15}
                    fontWeight="700"
                    pointerEvents="none"
                  >
                    {reviewed ? '✓' : '!'}
                  </text>
                )}

                {markerMode === 'label' && (
                  <g pointerEvents="none">
                    <rect
                      x={point.column + radius * 1.5}
                      y={point.row - radius}
                      width={Math.max(
                        radius * 5,
                        pretty(label).length * radius * 0.58
                      )}
                      height={radius * 2}
                      rx={radius * 0.35}
                      fill="#092839"
                      opacity={0.95}
                    />

                    <text
                      x={point.column + radius * 1.9}
                      y={point.row}
                      fill="white"
                      dominantBaseline="central"
                      fontSize={radius * 0.85}
                    >
                      {pretty(label)}
                    </text>
                  </g>
                )}
              </g>
            )
          })}
      </svg>
    )
  }

  function renderInspector(reviewMode: boolean) {
    if (!activePhoto?.analysis || !activePoint) {
      return null
    }

    const photo = activePhoto
    const point = activePoint
    const decision = photo.decisions[point.point]
    const draft = photo.drafts[point.point]

    const index = points.findIndex(
      item => item.point === point.point
    )

    const flagged = points.filter(
      item => item.needs_review
    ).length

    const completed = flagged - pending.length

    return (
      <section className="panel inspector">
        <div className="panel-heading">
          <h2>Point {point.point}</h2>

          <span
            className={`badge ${
              pointStatus(photo, point) === 'Needs review'
                ? 'warning'
                : decision?.reviewed
                  ? 'success'
                  : ''
            }`}
          >
            {pointStatus(photo, point)}
          </span>
        </div>

        <PointCrop photo={photo} point={point} />

        <p className="crop-caption">
          224 × 224 px view · Center marked
        </p>

        <div className="prediction-row">
          <div>
            <span className="muted">
              Original prediction
            </span>
            <strong>{pretty(point.prediction)}</strong>
          </div>

          <strong className="confidence">
            {percent(point.confidence)}
          </strong>
        </div>

        <p className="footnote">
          Model confidence is not measured accuracy.
        </p>

        <div className="point-navigation">
          <button
            className="button secondary"
            disabled={index <= 0}
            onClick={() =>
              setSelectedPointId(points[index - 1].point)
            }
          >
            Previous
          </button>

          <button
            className="button secondary"
            disabled={index >= points.length - 1}
            onClick={() =>
              setSelectedPointId(points[index + 1].point)
            }
          >
            Next
          </button>
        </div>

        <label className="field">
          <span>Review label</span>

          <select
            className="input"
            value={
              draft ??
              decision?.label ??
              point.prediction
            }
            disabled={busyPhotoId === photo.id}
            onChange={event =>
              chooseLabel(photo, point, event.target.value)
            }
          >
            {photo.analysis!.classes.map(label => (
              <option key={label} value={label}>
                {pretty(label)}
              </option>
            ))}
          </select>
        </label>

        {draft !== undefined && (
          <div className="draft-notice">
            <span>
              Unconfirmed selection. Current results
              are unchanged.
            </span>

            <button
              className="text-button"
              onClick={() => discardDraft(photo, point)}
            >
              Discard selection
            </button>
          </div>
        )}

        <button
          className="button primary full"
          disabled={
            busyPhotoId === photo.id ||
            (!!decision?.reviewed && draft === undefined)
          }
          onClick={() =>
            confirmPoint(photo, point, reviewMode)
          }
        >
          {decision?.reviewed && draft === undefined
            ? 'Review confirmed'
            : reviewMode
              ? 'Confirm & next pending'
              : 'Confirm review'}
        </button>

        {decision?.reviewed && (
          <p className="confirmed-label">
            Current confirmed label:{' '}
            <strong>{pretty(decision.label)}</strong>
          </p>
        )}

        <div className="review-progress">
          <div>
            <strong>
              {completed} / {flagged}
            </strong>
            <span>Flagged points reviewed</span>
          </div>
          <span>{pending.length} pending</span>
        </div>

        {flagged > 0 ? (
          <progress
            max={flagged}
            value={completed}
            aria-label="Flagged points reviewed"
          />
        ) : (
          <p className="footnote">
            No points were flagged. Any point can still
            be reviewed.
          </p>
        )}
      </section>
    )
  }

  function renderAnalysis(reviewMode: boolean) {
    if (!workspace.surveys.length) {
      return (
        <section className="panel empty">
          <h2>Start with a survey</h2>
          <p>
            Create a monitoring site and a survey,
            then upload your photographs.
          </p>

          <button
            className="button primary"
            onClick={() =>
              workspace.sites.length
                ? setShowSurveyForm(true)
                : setShowSiteForm(true)
            }
          >
            {workspace.sites.length
              ? 'Create survey'
              : 'Add monitoring site'}
          </button>
        </section>
      )
    }

    const average = points.length
      ? points.reduce(
          (sum, point) => sum + point.confidence,
          0
        ) / points.length
      : 0

    const rows = pendingOnly ? pending : points

    return (
      <>
        <div className="toolbar panel">
          <label className="toolbar-field">
            <span>Current survey</span>

            <select
              className="input"
              value={activeSurvey?.id ?? ''}
              onChange={event =>
                openSurvey(
                  event.target.value,
                  reviewMode ? 'review' : 'analysis'
                )
              }
            >
              {workspace.surveys.map(survey => {
                const site = workspace.sites.find(
                  item => item.id === survey.siteId
                )

                return (
                  <option key={survey.id} value={survey.id}>
                    {site?.name} — {survey.name}
                  </option>
                )
              })}
            </select>
          </label>

          <label className="button secondary upload-button">
            Add photographs
            <input
              type="file"
              multiple
              accept="image/jpeg,image/png,image/webp"
              onChange={event => {
                addPhotos(
                  Array.from(event.target.files ?? [])
                )
                event.target.value = ''
              }}
            />
          </label>

          <button
            className="button secondary"
            disabled={
              !surveyPhotos.some(photo => photo.analysis)
            }
            onClick={() =>
              exportPhotos(
                surveyPhotos,
                'ReefEye_survey_results.csv'
              )
            }
          >
            Export survey
          </button>
        </div>

        {!activePhoto ? (
          <section className="panel empty">
            <h2>This survey has no photographs yet</h2>
            <p>
              Use “Add photographs” to build its image album.
            </p>
          </section>
        ) : (
          <>
            <div className="analysis-metrics">
              <div>
                <span>Sample points</span>
                <strong>{points.length || '—'}</strong>
              </div>

              <div>
                <span>Mean confidence</span>
                <strong>
                  {points.length ? percent(average) : '—'}
                </strong>
              </div>

              <div>
                <span>Awaiting review</span>
                <strong>
                  {activePhoto.analysis ? pending.length : '—'}
                </strong>
              </div>

              <div>
                <span>Manually reviewed</span>
                <strong>{reviewedCount(activePhoto)}</strong>
              </div>
            </div>

            <div className="workspace-grid">
              <div className="workspace-main">
                <section className="panel image-panel">
                  <div className="panel-heading">
                    <div className="filename-heading">
                      <h2>{activePhoto.filename}</h2>
                      <p>
                        {surveyPhotos.findIndex(
                          photo => photo.id === activePhoto.id
                        ) + 1}
                        {' '}of {surveyPhotos.length} photographs
                      </p>
                    </div>

                    {activePhoto.analysis && (
                      <button
                        className="button secondary"
                        onClick={() =>
                          exportPhotos(
                            [activePhoto],
                            'ReefEye_image_results.csv'
                          )
                        }
                      >
                        Export image
                      </button>
                    )}
                  </div>

                  {activePhoto.analysis && (
                    <div className="image-controls">
                      <label className="field">
                        <span>Point labels</span>

                        <select
                          className="input"
                          value={markerMode}
                          onChange={event =>
                            setMarkerMode(
                              event.target.value as
                                | 'number'
                                | 'confidence'
                                | 'label'
                            )
                          }
                        >
                          <option value="number">
                            Point number
                          </option>
                          <option value="confidence">
                            Model confidence
                          </option>
                          <option value="label">
                            Classification name
                          </option>
                        </select>
                      </label>

                      <label className="field">
                        <span>Classification filter</span>

                        <select
                          className="input"
                          value={classFilter}
                          onChange={event =>
                            setClassFilter(event.target.value)
                          }
                        >
                          <option value="all">
                            All classifications
                          </option>

                          {activePhoto.analysis.classes.map(
                            label => (
                              <option key={label} value={label}>
                                {pretty(label)}
                              </option>
                            )
                          )}
                        </select>
                      </label>

                      <div className="checkbox-row">
                        <label>
                          <input
                            type="checkbox"
                            checked={showMarkers}
                            onChange={event =>
                              setShowMarkers(
                                event.target.checked
                              )
                            }
                          />
                          Show points
                        </label>

                        <label>
                          <input
                            type="checkbox"
                            checked={pendingOnly}
                            onChange={event =>
                              setPendingOnly(
                                event.target.checked
                              )
                            }
                          />
                          Pending only
                        </label>
                      </div>
                    </div>
                  )}

                  <div className="image-stage">
                    {renderImageCanvas()}
                  </div>

                  {activePhoto.analysis && (
                    <div className="image-legend">
                      {activePhoto.analysis.classes.map(label => (
                        <span className="color-label" key={label}>
                          <i
                            style={{
                              background:
                                COLORS[label] ?? '#475569',
                            }}
                          />
                          {pretty(label)}
                        </span>
                      ))}

                      <p>
                        ! Needs review · ✓ Reviewed ·
                        Outer ring: selected point
                      </p>
                    </div>
                  )}
                </section>

                <section className="panel album-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Survey album</h2>
                      <p>
                        Select a photograph to inspect
                        or analyze.
                      </p>
                    </div>
                  </div>

                  <div className="photo-album">
                    {surveyPhotos.map(photo => (
                      <button
                        key={photo.id}
                        className={`album-item ${
                          photo.id === activePhoto.id
                            ? 'selected'
                            : ''
                        }`}
                        onClick={() => selectPhoto(photo.id)}
                        aria-pressed={
                          photo.id === activePhoto.id
                        }
                      >
                        <PhotoImage
                          photo={photo}
                          className="album-image"
                        />

                        <strong title={photo.filename}>
                          {photo.filename}
                        </strong>

                        <span>
                          {busyPhotoId === photo.id
                            ? 'Analyzing…'
                            : !photo.analysis
                              ? 'Not analyzed'
                              : pendingPoints(photo).length
                                ? `${pendingPoints(photo).length} pending`
                                : 'No pending flags'}
                        </span>
                      </button>
                    ))}
                  </div>
                </section>

                {activePhoto.analysis && (
                  <Distribution photos={[activePhoto]} />
                )}
              </div>

              <aside className="workspace-side">
                <section className="panel">
                  <div className="panel-heading">
                    <h2>Survey information</h2>
                  </div>

                  <dl className="metadata">
                    <div>
                      <dt>Site</dt>
                      <dd>{activeSite?.name ?? '—'}</dd>
                    </div>

                    <div>
                      <dt>Survey</dt>
                      <dd>{activeSurvey?.name}</dd>
                    </div>

                    <div>
                      <dt>Date</dt>
                      <dd>
                        {formatDate(activeSurvey?.date ?? '')}
                      </dd>
                    </div>

                    <div>
                      <dt>Depth</dt>
                      <dd>
                        {activeSurvey?.depth != null
                          ? `${activeSurvey.depth} m`
                          : 'Not recorded'}
                      </dd>
                    </div>

                    <div>
                      <dt>Visibility</dt>
                      <dd>
                        {activeSurvey?.visibility != null
                          ? `${activeSurvey.visibility} m`
                          : 'Not recorded'}
                      </dd>
                    </div>

                    <div>
                      <dt>Photographs</dt>
                      <dd>{surveyPhotos.length}</dd>
                    </div>
                  </dl>

                  {activeSurvey?.notes && (
                    <p className="survey-notes">
                      {activeSurvey.notes}
                    </p>
                  )}
                </section>

                {!reviewMode && (
                  <section className="panel">
                    <h2>Analysis settings</h2>

                    <label className="field">
                      <span>
                        Sampling grid for next analysis
                      </span>

                      <select
                        className="input"
                        value={activePhoto.gridSize}
                        disabled={busyPhotoId !== null}
                        onChange={event => {
                          const gridSize = Number(
                            event.target.value
                          )

                          updatePhoto(
                            activePhoto.id,
                            photo => ({
                              ...photo,
                              gridSize,
                            })
                          )
                        }}
                      >
                        <option value={5}>
                          5 × 5 · Up to 25 points
                        </option>
                        <option value={7}>
                          7 × 7 · Up to 49 points
                        </option>
                        <option value={10}>
                          10 × 10 · Up to 100 points
                        </option>
                      </select>
                    </label>

                    <button
                      className={`button ${
                        activePhoto.analysis
                          ? 'secondary'
                          : 'primary'
                      } full`}
                      disabled={busyPhotoId !== null}
                      onClick={() => analyzePhoto(activePhoto)}
                    >
                      {busyPhotoId === activePhoto.id
                        ? 'Analyzing…'
                        : activePhoto.analysis
                          ? 'Reanalyze photograph'
                          : 'Analyze photograph'}
                    </button>

                    {busyPhotoId === activePhoto.id && (
                      <button
                        className="text-button"
                        onClick={() =>
                          controllerRef.current?.abort()
                        }
                      >
                        Cancel analysis
                      </button>
                    )}
                  </section>
                )}

                {activePhoto.analysis &&
                  pending.length > 0 &&
                  !reviewMode && (
                    <section className="panel review-callout">
                      <h2>
                        {pending.length} points need review
                      </h2>

                      <p>
                        Inspect the image crops before
                        confirming their labels.
                      </p>

                      <button
                        className="button coral full"
                        onClick={() => {
                          setPendingOnly(true)
                          setSelectedPointId(pending[0].point)
                          setPage('review')
                        }}
                      >
                        Start human review
                      </button>
                    </section>
                  )}

                {renderInspector(reviewMode)}

                {reviewMode && !activePhoto.analysis && (
                  <section className="panel empty compact">
                    <h2>Analyze this photograph first</h2>

                    <button
                      className="button primary"
                      onClick={() => navigate('analysis')}
                    >
                      Open AI Analysis
                    </button>
                  </section>
                )}
              </aside>
            </div>

            {reviewMode && activePhoto.analysis && (
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Point review table</h2>
                    <p>
                      Selections update current labels
                      only after confirmation.
                    </p>
                  </div>

                  <label className="check-label">
                    <input
                      type="checkbox"
                      checked={pendingOnly}
                      onChange={event =>
                        setPendingOnly(event.target.checked)
                      }
                    />
                    Pending points only
                  </label>
                </div>

                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Point</th>
                        <th scope="col">Original prediction</th>
                        <th scope="col">Confidence</th>
                        <th scope="col">Status</th>
                        <th scope="col">Review label</th>
                        <th scope="col">Action</th>
                      </tr>
                    </thead>

                    <tbody>
                      {rows.map(point => {
                        const decision =
                          activePhoto.decisions[point.point]
                        const draft =
                          activePhoto.drafts[point.point]

                        return (
                          <tr
                            key={point.point}
                            className={
                              activePoint?.point === point.point
                                ? 'selected-row'
                                : ''
                            }
                          >
                            <td>
                              <button
                                className="text-button"
                                onClick={() =>
                                  setSelectedPointId(point.point)
                                }
                              >
                                Point {point.point}
                              </button>
                            </td>

                            <td>
                              {pretty(point.prediction)}

                              {decision?.reviewed && (
                                <small className="current-label">
                                  Current:{' '}
                                  {pretty(decision.label)}
                                </small>
                              )}
                            </td>

                            <td>{percent(point.confidence)}</td>

                            <td>
                              {pointStatus(activePhoto, point)}
                            </td>

                            <td>
                              <select
                                className="input"
                                aria-label={`Review label for point ${point.point}`}
                                value={
                                  draft ??
                                  decision?.label ??
                                  point.prediction
                                }
                                disabled={
                                  busyPhotoId === activePhoto.id
                                }
                                onChange={event =>
                                  chooseLabel(
                                    activePhoto,
                                    point,
                                    event.target.value
                                  )
                                }
                              >
                                {activePhoto.analysis!.classes.map(
                                  label => (
                                    <option
                                      key={label}
                                      value={label}
                                    >
                                      {pretty(label)}
                                    </option>
                                  )
                                )}
                              </select>
                            </td>

                            <td>
                              <button
                                className="button secondary"
                                disabled={
                                  busyPhotoId === activePhoto.id ||
                                  (!!decision?.reviewed &&
                                    draft === undefined)
                                }
                                onClick={() =>
                                  confirmPoint(
                                    activePhoto,
                                    point,
                                    false
                                  )
                                }
                              >
                                {decision?.reviewed &&
                                draft === undefined
                                  ? 'Confirmed'
                                  : 'Confirm'}
                              </button>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>

                {!rows.length && (
                  <div className="empty compact">
                    No pending points. Turn off the filter
                    to inspect all predictions.
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </>
    )
  }

  function renderDirectory() {
    const counts = labelCounts(workspace.photos)

    return (
      <>
        <section className="panel directory-intro">
          <h2>The labels behind your observations</h2>
          <p>
            Examples below come from your analyzed photographs.
            A model prediction is not an independent biological
            identification.
          </p>
        </section>

        <div className="directory-grid">
          {MODEL_CLASSES.map(label => {
            let example:
              | { photo: Photo; point: Point }
              | undefined

            for (const photo of analyzedPhotos) {
              const point = photo.analysis!.points.find(
                item => currentLabel(photo, item) === label
              )

              if (point) {
                example = { photo, point }
                break
              }
            }

            return (
              <article
                className="panel directory-card"
                key={label}
              >
                {example ? (
                  <PointCrop
                    photo={example.photo}
                    point={example.point}
                  />
                ) : (
                  <div
                    className="directory-placeholder"
                    style={{ borderColor: COLORS[label] }}
                  >
                    <span
                      style={{ background: COLORS[label] }}
                    />
                    {pretty(label)}
                  </div>
                )}

                <div className="card-title-row">
                  <h2>{pretty(label)}</h2>
                  <i
                    className="class-dot"
                    style={{ background: COLORS[label] }}
                  />
                </div>

                <p>
                  {label === 'background'
                    ? 'Background classification'
                    : label === 'other_coral'
                      ? 'Other coral classification'
                      : 'Coral genus classification'}
                </p>

                <strong>
                  {counts[label] ?? 0} sampled points
                </strong>

                {example ? (
                  <p className="footnote">
                    Example:{' '}
                    {pointStatus(
                      example.photo,
                      example.point
                    )}{' '}
                    · {percent(example.point.confidence)}{' '}
                    original confidence
                  </p>
                ) : (
                  <p className="footnote">
                    No example in your workspace yet.
                  </p>
                )}
              </article>
            )
          })}
        </div>
      </>
    )
  }

  function renderExports() {
    return (
      <>
        <section className="panel export-overview">
          <div>
            <span className="eyebrow">
              WORKSPACE DATA
            </span>
            <h2>{totalPoints} sampled points</h2>
            <p>
              Across {analyzedPhotos.length} analyzed
              photographs and {workspace.surveys.length} surveys.
            </p>
          </div>

          <button
            className="button primary"
            disabled={!analyzedPhotos.length}
            onClick={() =>
              exportPhotos(
                workspace.photos,
                'ReefEye_workspace_results.csv'
              )
            }
          >
            Export all results
          </button>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Export by survey</h2>
              <p>
                CSV includes all analyzed points, original
                predictions, confirmed labels and recorded
                survey metadata.
              </p>
            </div>
          </div>

          <div className="export-list">
            {workspace.surveys.map(survey => {
              const photos = workspace.photos.filter(
                photo => photo.surveyId === survey.id
              )

              const analyzed = photos.filter(
                photo => photo.analysis
              ).length

              const site = workspace.sites.find(
                item => item.id === survey.siteId
              )

              return (
                <div
                  className="export-item"
                  key={survey.id}
                >
                  <div>
                    <strong>{survey.name}</strong>
                    <span>
                      {site?.name} · {formatDate(survey.date)}
                    </span>
                    <small>
                      {analyzed} analyzed / {photos.length}{' '}
                      photographs
                    </small>
                  </div>

                  <button
                    className="button secondary"
                    disabled={!analyzed}
                    onClick={() =>
                      exportPhotos(
                        photos,
                        'ReefEye_survey_results.csv'
                      )
                    }
                  >
                    Export CSV
                  </button>
                </div>
              )
            })}
          </div>

          {!workspace.surveys.length && (
            <div className="empty compact">
              No survey data to export yet.
            </div>
          )}

          <p className="footnote">
            CSV exports exclude image files and unconfirmed
            selections. Browser storage is local to this device
            and website address.
          </p>
        </section>
      </>
    )
  }

  return (
    <div className="reefeye-app" lang="en" dir="ltr">
      <aside className="sidebar">
        <div className="brand">
          <img src="/logo.png" alt="ReefEye logo" />
          <div>
            <strong>REEFEYE</strong>
            <span>Reef Observatory</span>
          </div>
        </div>

        <div className="nav-caption">WORKSPACE</div>

        <nav aria-label="Main navigation">
          {NAV.map((item, index) => (
            <button
              key={item.id}
              className={`nav-item ${
                page === item.id ? 'active' : ''
              }`}
              aria-current={
                page === item.id ? 'page' : undefined
              }
              onClick={() => {
                if (item.id === 'review') {
                  setPendingOnly(true)
                }
                navigate(item.id)
              }}
            >
              <span
                className="nav-number"
                aria-hidden="true"
              >
                {String(index + 1).padStart(2, '0')}
              </span>

              <span>{item.label}</span>

              {item.id === 'review' && totalPending > 0 && (
                <span className="nav-count">
                  {totalPending}
                </span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <span className="status-indicator">
            <i
              className={
                connection === 'Model connected'
                  ? 'connected'
                  : ''
              }
            />
            {connection}
          </span>

          <p>
            Coral reef observations
            <br />
            Red Sea · Saudi Arabia
          </p>
        </div>
      </aside>

      <main className="main-content">
        <header className="page-header">
          <div>
            <span className="eyebrow">
              REEFEYE / WORKSPACE
            </span>
            <h1>{PAGE_INFO[page].title}</h1>
            <p>{PAGE_INFO[page].description}</p>
          </div>

          <div className="header-actions">
            <button
              className="button secondary"
              disabled={!ready}
              onClick={() => setShowSiteForm(true)}
            >
              Add site
            </button>

            <button
              className="button primary"
              disabled={!ready || !workspace.sites.length}
              onClick={() => setShowSurveyForm(true)}
            >
              New survey
            </button>
          </div>
        </header>

        <div className="workspace-status">
          <span role="status">{saveStatus}</span>
          <span>
            {workspace.surveys.length} surveys ·{' '}
            {workspace.photos.length} images
          </span>
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button
              className="text-button"
              onClick={() => setError('')}
            >
              Dismiss
            </button>
          </div>
        )}

        {(message || undo) && (
          <div
            className="message-banner"
            role="status"
            aria-live="polite"
          >
            <span>{message}</span>

            {undo && (
              <button
                className="text-button"
                onClick={undoLastReview}
              >
                Undo last review
              </button>
            )}
          </div>
        )}

        {showSiteForm && ready && (
          <section className="panel form-panel">
            <div className="panel-heading">
              <div>
                <h2>Add monitoring site</h2>
                <p>
                  Coordinates are optional and must describe
                  the actual site.
                </p>
              </div>

              <button
                className="text-button"
                onClick={() => setShowSiteForm(false)}
              >
                Close
              </button>
            </div>

            <form
              className="form-grid"
              onSubmit={createSite}
            >
              <label className="field wide">
                <span>Site name</span>
                <input
                  className="input"
                  name="name"
                  required
                  maxLength={100}
                  placeholder="e.g. Al-Wajh"
                />
              </label>

              <label className="field">
                <span>Latitude · optional</span>
                <input
                  className="input"
                  name="latitude"
                  type="number"
                  step="any"
                  min={-85}
                  max={85}
                  placeholder="Decimal degrees"
                />
              </label>

              <label className="field">
                <span>Longitude · optional</span>
                <input
                  className="input"
                  name="longitude"
                  type="number"
                  step="any"
                  min={-180}
                  max={180}
                  placeholder="Decimal degrees"
                />
              </label>

              <div className="form-actions wide">
                <button
                  className="button primary"
                  type="submit"
                >
                  Save monitoring site
                </button>
              </div>
            </form>
          </section>
        )}

        {showSurveyForm && ready && (
          <section className="panel form-panel">
            <div className="panel-heading">
              <div>
                <h2>Create survey</h2>
                <p>
                  Group photographs from the same site visit.
                </p>
              </div>

              <button
                className="text-button"
                onClick={() => {
                  setShowSurveyForm(false)
                  setNewSurveyFiles([])
                }}
              >
                Close
              </button>
            </div>

            <form
              className="form-grid"
              onSubmit={createSurvey}
            >
              <label className="field">
                <span>Monitoring site</span>
                <select
                  className="input"
                  name="siteId"
                  required
                  defaultValue=""
                >
                  <option value="" disabled>
                    Choose site
                  </option>

                  {workspace.sites.map(site => (
                    <option key={site.id} value={site.id}>
                      {site.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Survey name</span>
                <input
                  className="input"
                  name="name"
                  required
                  maxLength={120}
                  placeholder="e.g. Northern reef survey"
                />
              </label>

              <label className="field">
                <span>Survey date</span>
                <input
                  className="input"
                  name="date"
                  type="date"
                  required
                  defaultValue={today()}
                />
              </label>

              <label className="field">
                <span>Depth in meters · optional</span>
                <input
                  className="input"
                  name="depth"
                  type="number"
                  min={0}
                  step="any"
                />
              </label>

              <label className="field">
                <span>
                  Water visibility in meters · optional
                </span>
                <input
                  className="input"
                  name="visibility"
                  type="number"
                  min={0}
                  step="any"
                />
              </label>

              <label className="field">
                <span>Photographs · optional</span>
                <input
                  className="input file-input"
                  type="file"
                  multiple
                  accept="image/jpeg,image/png,image/webp"
                  onChange={event =>
                    setNewSurveyFiles(
                      Array.from(event.target.files ?? [])
                    )
                  }
                />
                <small>
                  JPG, PNG or WebP · Maximum 10 MB per image
                </small>
              </label>

              <label className="field wide">
                <span>Survey notes · optional</span>
                <textarea
                  className="input"
                  name="notes"
                  rows={3}
                  maxLength={2000}
                  placeholder="Record relevant field observations."
                />
              </label>

              <div className="form-actions wide">
                <span className="muted">
                  {newSurveyFiles.length} photographs selected
                </span>

                <button
                  className="button primary"
                  type="submit"
                  disabled={!workspace.sites.length}
                >
                  Create survey
                </button>
              </div>
            </form>
          </section>
        )}

        {!ready ? (
          <section className="panel empty">
            Opening your saved workspace…
          </section>
        ) : (
          <>
            {page === 'dashboard' && renderDashboard()}
            {page === 'surveys' && renderSurveys()}
            {page === 'sites' && renderSites()}
            {page === 'analysis' && renderAnalysis(false)}
            {page === 'review' && renderAnalysis(true)}
            {page === 'directory' && renderDirectory()}
            {page === 'exports' && renderExports()}
          </>
        )}
      </main>
    </div>
  )
}

export default App