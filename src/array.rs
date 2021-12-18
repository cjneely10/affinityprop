#[derive(Debug)]
pub struct Array<T> {
    vec: Vec<T>,
    row_size: usize,
}

impl<T> Array<T>
where
    T: Default + Copy,
{
    pub fn new(r: usize, c: usize) -> Self {
        Self {
            vec: vec![T::default(); r * c],
            row_size: r,
        }
    }

    pub fn rows(&self) -> usize {
        self.row_size
    }

    /// Get value at index in array
    pub fn at(&self, i: usize) -> Option<&T> {
        self.vec.get(i)
    }

    /// Get value at (r,c) index in array
    pub fn get(&self, r: usize, c: usize) -> Option<&T> {
        self.vec.get(r * self.row_size + c)
    }

    pub fn set(&mut self, r: usize, c: usize, val: T) {
        self.vec[r * self.row_size + c] = val;
    }
}

impl<T> From<&Array<T>> for Array<T>
where
    T: Default + Copy,
{
    fn from(f: &Array<T>) -> Self {
        Self {
            vec: f.vec.clone(),
            row_size: f.row_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let mut array = Array::new(1000, 1000);
        let mut counter = 0;
        (0..1000)
            .into_iter()
            .map(|a| {
                (0..1000)
                    .into_iter()
                    .map(|b| {
                        array.set(a, b, counter);
                        counter += 1;
                    })
                    .last();
            })
            .last();
        (0..1000)
            .into_iter()
            .rev()
            .map(|a| {
                (0..1000)
                    .into_iter()
                    .rev()
                    .map(|b| {
                        let val = array.get(a, b);
                        assert_eq!(val.unwrap(), &(counter - 1));
                        counter -= 1;
                    })
                    .last();
            })
            .last();
    }

    #[test]
    #[should_panic]
    fn oob() {
        let mut array: Array<u8> = Array::new(10, 10);
        assert_eq!(None, array.get(11, 9));
        array.set(11, 9, 10);
    }

    #[test]
    fn at_get() {
        let mut array: Array<u8> = Array::new(10, 10);
        array.set(5, 5, 1);
        assert_eq!(array.get(5, 5), array.at(5 * 10 + 5));
    }
}
