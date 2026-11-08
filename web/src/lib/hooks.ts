import { useEffect, useState } from 'react'
import { getMeta, type Meta } from '../api'

export function useMediaQuery(query: string): boolean {
  const [match, setMatch] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches)
  useEffect(() => {
    const m = window.matchMedia(query)
    const on = () => setMatch(m.matches)
    on()
    m.addEventListener('change', on)
    return () => m.removeEventListener('change', on)
  }, [query])
  return match
}

let metaCache: Meta | null = null

export function useMeta(): { meta: Meta | null; failed: boolean } {
  const [meta, setMeta] = useState<Meta | null>(metaCache)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    if (metaCache) return
    const c = new AbortController()
    getMeta(c.signal)
      .then((m) => {
        metaCache = m
        setMeta(m)
      })
      .catch((e) => {
        if ((e as Error).name !== 'AbortError') setFailed(true)
      })
    return () => c.abort()
  }, [])
  return { meta, failed }
}

/** "/" focuses the question box from anywhere that is not already a text field. */
export function useSlashFocus(target: React.RefObject<HTMLTextAreaElement | null>) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return
      const el = document.activeElement
      if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || (el as HTMLElement)?.isContentEditable) return
      e.preventDefault()
      target.current?.focus()
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [target])
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
