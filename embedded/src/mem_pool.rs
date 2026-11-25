//! Vung nho chung P/Q/R theo Bang 2 de cuong, co san r va thu ho co thu tu.
//! Khac sample_queue.rs: co R, co log ly do bo mau, khong han chong trung o ID >= 32.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CellState { Free, Pending, Ready, Replay }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DropReason { PoolFull }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LabelError { NotFound, AlreadyLabeled, AlreadyLearned, EvictedMiss }

#[derive(Clone, Copy)]
pub struct Cell {
    pub id: u32,
    pub label: i16,          // -1: chua co nhan
    pub z: [i8; 16],
    pub state: CellState,
    pub age: u64,            // thoi diem gan vao o gan nhat
}

const EMPTY: Cell = Cell { id: 0, label: -1, z: [0; 16], state: CellState::Free, age: 0 };

pub struct MemPool<const K: usize> {
    cells: [Cell; K],
    r_floor: usize,          // r: so mau R toi thieu duoc bao ve
    clock: u64,
    next_id: u32,
    pub pushed: u32,
    pub matched: u32,
    pub miss_labels: u32,    // nhan ve sau khi o bi thu hoi (ghep truot)
    pub learned_first: u32,
    pub replay_learns: u32,
    pub recovered_p: u32,
    pub recovered_r: u32,
    pub dup_labels: u32,
    last_replayed: u32,      // ID vua hoc luot lien truoc (bo qua khi quay so)
}

impl<const K: usize> MemPool<K> {
    pub const fn new(r_floor: usize) -> Self {
        Self {
            cells: [EMPTY; K], r_floor, clock: 0, next_id: 0,
            pushed: 0, matched: 0, miss_labels: 0, learned_first: 0,
            replay_learns: 0, recovered_p: 0, recovered_r: 0, dup_labels: 0,
            last_replayed: u32::MAX,
        }
    }

    pub fn cell_size_bytes() -> usize { core::mem::size_of::<Cell>() }

    fn count(&self, st: CellState) -> usize {
        let mut n = 0;
        for c in self.cells.iter() {
            if c.state == st { n += 1; }
        }
        n
    }

    /// Thu tu thu ho (de cuong muc 3): o trong -> R cu nhat khi R > r -> P cho lau nhat.
    /// Q duoc bao ve, khong bao gio bi duoi.
    fn reclaim_one(&mut self) -> Option<usize> {
        let mut free: Option<usize> = None;
        let mut old_r: Option<usize> = None;
        let mut old_p: Option<usize> = None;
        let mut i = 0;
        while i < K {
            match self.cells[i].state {
                CellState::Free => { if free.is_none() { free = Some(i); } }
                CellState::Replay => {
                    if let Some(j) = old_r {
                        if self.cells[i].age < self.cells[j].age { old_r = Some(i); }
                    } else { old_r = Some(i); }
                }
                CellState::Pending => {
                    if let Some(j) = old_p {
                        if self.cells[i].age < self.cells[j].age { old_p = Some(i); }
                    } else { old_p = Some(i); }
                }
                CellState::Ready => {}
            }
            i += 1;
        }
        if let Some(j) = free { return Some(j); }
        if self.count(CellState::Replay) > self.r_floor {
            if let Some(j) = old_r {
                self.recovered_r += 1;
                self.cells[j] = EMPTY;
                return Some(j);
            }
        }
        if let Some(j) = old_p {
            self.recovered_p += 1;
            self.cells[j] = EMPTY;
            return Some(j);
        }
        None // chi con Q day -> PoolFull
    }

    pub fn push(&mut self, z: [i8; 16]) -> Result<u32, DropReason> {
        self.clock += 1;
        let id = self.next_id;
        let slot = match self.reclaim_one() {
            Some(j) => j,
            None => return Err(DropReason::PoolFull),
        };
        self.cells[slot] = Cell { id, label: -1, z, state: CellState::Pending, age: self.clock };
        self.next_id += 1;
        self.pushed += 1;
        Ok(id)
    }

    pub fn on_label(&mut self, id: u32, label: u8) -> Result<(), LabelError> {
        let mut found: Option<usize> = None;
        let mut i = 0;
        while i < K {
            if self.cells[i].id == id && self.cells[i].state != CellState::Free {
                found = Some(i);
                break;
            }
            i += 1;
        }
        match found {
            None => {
                self.miss_labels += 1;
                Err(LabelError::EvictedMiss)
            }
            Some(j) => match self.cells[j].state {
                CellState::Pending => {
                    self.cells[j].label = label as i16;
                    self.cells[j].state = CellState::Ready;
                    self.matched += 1;
                    Ok(())
                }
                CellState::Ready => { self.dup_labels += 1; Err(LabelError::AlreadyLabeled) }
                CellState::Replay => { self.dup_labels += 1; Err(LabelError::AlreadyLearned) }
                CellState::Free => unreachable!(),
            },
        }
    }

    /// Lay mau co age nho nhat o Q (FIFO that, khong theo vi tri vong).
    pub fn pop_ready(&mut self) -> Option<Cell> {
        let mut best: Option<usize> = None;
        let mut i = 0;
        while i < K {
            if self.cells[i].state == CellState::Ready {
                if let Some(j) = best {
                    if self.cells[i].age < self.cells[j].age { best = Some(i); }
                } else { best = Some(i); }
            }
            i += 1;
        }
        let j = best?;
        let c = self.cells[j];
        self.cells[j] = EMPTY;
        self.learned_first += 1;
        Some(c)
    }

    /// Sau lan hoc dau: chuyen o sang R. Khong con o trong -> mat lan hoc lai, van tinh da hoc dau.
    pub fn promote_to_replay(&mut self, c: Cell) -> bool {
        self.clock += 1;
        let mut i = 0;
        while i < K {
            if self.cells[i].state == CellState::Free {
                self.cells[i] = Cell { state: CellState::Replay, age: self.clock, ..c };
                return true;
            }
            i += 1;
        }
        false
    }

    /// Quay so deu trong R, bo qua ID vua hoc luot lien truoc (de cuong muc 5).
    pub fn replay_draw(&mut self, rnd: &mut u32) -> Option<Cell> {
        let n = self.count(CellState::Replay);
        if n == 0 { return None; }
        let mut k = *rnd;
        k ^= k << 13; k ^= k >> 17; k ^= k << 5;
        *rnd = k;
        let pick = (k as usize) % n;
        let mut seen = 0usize;
        let mut i = 0;
        while i < K {
            if self.cells[i].state == CellState::Replay {
                if seen == pick && self.cells[i].id != self.last_replayed {
                    self.replay_learns += 1;
                    self.last_replayed = self.cells[i].id;
                    return Some(self.cells[i]);
                }
                seen += 1;
            }
            i += 1;
        }
        None
    }
}
