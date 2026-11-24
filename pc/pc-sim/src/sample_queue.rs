//! Quan ly hang doi mau P (cho nhan) va Q (cho hoc lan dau).
//! Module thuan no_std, khong dung alloc, kich thuoc co dinh theo const generic.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleState {
    Pending,   // P: mau da phat, chua co nhan
    Ready,     // Q: da ghep nhan, cho hoc lan dau
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PushError {
    Full,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResolveError {
    NotFound,        // Khong tim thay ID trong P
    AlreadyLabeled,  // Mau dang o Q da co nhan
    AlreadyLearned,  // Mau da hoc xong
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub id: u32,
    pub label: Option<u8>,
    pub features: [i8; 16],
    pub state: SampleState,
}

impl Sample {
    pub const fn new_pending(id: u32, features: [i8; 16]) -> Self {
        Self { id, label: None, features, state: SampleState::Pending }
    }
}

/// Hang doi vong co dinh, khong alloc.
pub struct FixedQueue<const N: usize> {
    buf: [Option<Sample>; N],
    head: usize,
    len: usize,
}

impl<const N: usize> FixedQueue<N> {
    pub const fn new() -> Self {
        Self { buf: [None; N], head: 0, len: 0 }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_full(&self) -> bool { self.len == N }

    pub fn push(&mut self, s: Sample) -> Result<(), PushError> {
        if self.len == N {
            return Err(PushError::Full);
        }
        let tail = (self.head + self.len) % N;
        self.buf[tail] = Some(s);
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<Sample> {
        if self.len == 0 {
            return None;
        }
        let s = self.buf[self.head].take();
        self.head = (self.head + 1) % N;
        self.len -= 1;
        s
    }

    pub fn find(&self, id: u32) -> Option<&Sample> {
        for i in 0..self.len {
            let idx = (self.head + i) % N;
            if let Some(ref s) = self.buf[idx] {
                if s.id == id {
                    return Some(s);
                }
            }
        }
        None
    }

    /// Xoa mau theo ID va dich cac phan tu phia sau de giu thu tu FIFO.
    fn remove_by_id(&mut self, id: u32) -> Option<Sample> {
        if self.len == 0 {
            return None;
        }
        let mut pos: Option<usize> = None;
        for i in 0..self.len {
            let idx = (self.head + i) % N;
            if let Some(ref s) = self.buf[idx] {
                if s.id == id {
                    pos = Some(i);
                    break;
                }
            }
        }
        let pos = pos?;
        let abs = (self.head + pos) % N;
        let s = self.buf[abs].take();
        
        // Dich chuyen cac phan tu phia sau ve truoc 1 o
        for i in pos..(self.len - 1) {
            let cur = (self.head + i) % N;
            let next = (self.head + i + 1) % N;
            self.buf[cur] = self.buf[next].take();
        }
        let last = (self.head + self.len - 1) % N;
        self.buf[last] = None;
        self.len -= 1;
        s
    }
}

/// Bo quan ly hai hang doi P, Q cung bo dem thong ke.
pub struct SampleManager<const P: usize, const Q: usize> {
    pub pending: FixedQueue<P>,
    pub ready: FixedQueue<Q>,
    pub pushed: u32,
    pub matched: u32,
    pub learned_first: u32,
    pub dropped_p_full: u32,
    pub dropped_q_full: u32,
    pub dup_labels: u32,
    pub orphan_labels: u32,
    completed_mask: u32, // Danh dau cac ID (0-31) da hoc xong
}

impl<const P: usize, const Q: usize> SampleManager<P, Q> {
    pub const fn new() -> Self {
        Self {
            pending: FixedQueue::new(),
            ready: FixedQueue::new(),
            pushed: 0,
            matched: 0,
            learned_first: 0,
            dropped_p_full: 0,
            dropped_q_full: 0,
            dup_labels: 0,
            orphan_labels: 0,
            completed_mask: 0,
        }
    }

    pub fn push_sample(&mut self, features: [i8; 16]) -> Result<u32, PushError> {
        let id = self.pushed;
        let s = Sample::new_pending(id, features);
        match self.pending.push(s) {
            Ok(()) => {
                self.pushed += 1;
                Ok(id)
            }
            Err(e) => {
                self.dropped_p_full += 1;
                Err(e)
            }
        }
    }

    pub fn resolve_label(&mut self, id: u32, label: u8) -> Result<(), ResolveError> {
        // 1. Kiem tra xem da hoc xong chua (tranh nhan trung sau khi hoc)
        if id < 32 && (self.completed_mask & (1u32 << id)) != 0 {
            self.dup_labels += 1;
            return Err(ResolveError::AlreadyLearned);
        }
        
        // 2. Kiem tra trong P
        match self.pending.find(id) {
            Some(s) if s.label.is_some() => {
                self.dup_labels += 1;
                return Err(ResolveError::AlreadyLabeled);
            }
            Some(_) => {}
            None => {
                // Khong o P, co the dang o Q
                if self.ready.find(id).is_some() {
                    self.dup_labels += 1;
                    return Err(ResolveError::AlreadyLabeled);
                }
                // Khong o P, khong o Q -> Nhan mo coi (Orphan)
                self.orphan_labels += 1;
                return Err(ResolveError::NotFound);
            }
        }
        
        // 3. Rut mau khoi P va day sang Q
        let mut s = self.pending.remove_by_id(id).unwrap();
        s.label = Some(label);
        s.state = SampleState::Ready;
        
        match self.ready.push(s) {
            Ok(()) => {
                self.matched += 1;
                Ok(())
            }
            Err(_) => {
                // Ghep nhan thanh cong nhung Q day -> bi loai
                self.dropped_q_full += 1;
                self.matched += 1;
                Ok(())
            }
        }
    }

    pub fn pop_for_learning(&mut self) -> Option<Sample> {
        let s = self.ready.pop();
        if let Some(ref smp) = s {
            self.learned_first += 1;
            if smp.id < 32 {
                self.completed_mask |= 1u32 << smp.id;
            }
        }
        s
    }
}
